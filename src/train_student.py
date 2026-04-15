import os
import math
import torch
import argparse
import numpy as np
import json
import time
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.optim import Adam
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.transforms import Distance
from torch_geometric.nn import global_mean_pool, global_add_pool, radius
from torch_geometric.utils import remove_self_loops
from datetime import datetime

from model import MXMNet, Config
from pamnet import PAMNet
from mixed_body_dataset import ApplicationDS
from student.model_student import get_student_model

HARTREE_TO_KCAL_MOL = 627.5095

# PAMNet wrapper: exposes node-level features for distillation\
class PAMNetWithFeatures(nn.Module):
    def __init__(self, pamnet):
        super().__init__()
        self.pamnet = pamnet
        
        # Copy attributes from the original PAMNet model
        self.dataset = pamnet.dataset
        self.dim = pamnet.dim
        self.n_layer = pamnet.n_layer
        self.cutoff_l = pamnet.cutoff_l
        self.cutoff_g = pamnet.cutoff_g
    
    def forward(self, data):
        # Standard forward pass matching the original PAMNet interface
        return self.pamnet(data)
    
    def _get_edge_info(self, edge_index, pos):
        edge_index, _ = remove_self_loops(edge_index)
        j, i = edge_index
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()
        return edge_index, dist

    # Returns: out - [batch_size], node_feat - [num_nodes, dim], graph_feat - [batch_size, dim]
    def forward_with_features(self, data, layer_idx=-1):
        pamnet = self.pamnet
        x_raw = data.x
        batch = data.batch

        # Handle different input formats 
        if x_raw.dim() == 1:
            x = torch.index_select(pamnet.embeddings, 0, x_raw.long())
        elif x_raw.shape[-1] == 1:
            x = torch.index_select(pamnet.embeddings, 0, x_raw.squeeze(-1).long())
        else:
            # Assume first column is atom type
            x = torch.index_select(pamnet.embeddings, 0, x_raw[:, 0].long())
        
        edge_index_l = data.edge_index
        pos = data.pos

        # Compute pairwise distances in global layer
        row, col = radius(pos, pos, pamnet.cutoff_g, batch, batch, max_num_neighbors=1000)
        edge_index_g = torch.stack([row, col], dim=0)
        edge_index_g, dist_g = self._get_edge_info(edge_index_g, pos)

        # Compute pairwise distances in local layer
        edge_index_l, dist_l = self._get_edge_info(edge_index_l, pos)
        
        # Compute indices for message passing
        idx_i, idx_j, idx_k, idx_kj, idx_ji, idx_i_pair, idx_j1_pair, idx_j2_pair, idx_jj_pair, idx_ji_pair = \
            pamnet.indices(edge_index_l, num_nodes=x.size(0))
        
        # Compute two-hop angles
        pos_ji, pos_kj = pos[idx_j] - pos[idx_i], pos[idx_k] - pos[idx_j]
        a = (pos_ji * pos_kj).sum(dim=-1)
        b = torch.linalg.cross(pos_ji, pos_kj).norm(dim=-1)
        angle2 = torch.atan2(b, a)

        # Compute one-hop angles
        pos_i_pair = pos[idx_i_pair]
        pos_j1_pair = pos[idx_j1_pair]
        pos_j2_pair = pos[idx_j2_pair]
        pos_ji_pair, pos_jj_pair = pos_j1_pair - pos_i_pair, pos_j2_pair - pos_j1_pair
        a = (pos_ji_pair * pos_jj_pair).sum(dim=-1)
        b = torch.linalg.cross(pos_ji_pair, pos_jj_pair).norm(dim=-1)
        angle1 = torch.atan2(b, a)

        # Get rbf and sbf embeddings
        rbf_l = pamnet.rbf_l(dist_l)
        rbf_g = pamnet.rbf_g(dist_g)
        sbf1 = pamnet.sbf(dist_l, angle1, idx_jj_pair)
        sbf2 = pamnet.sbf(dist_l, angle2, idx_kj)

        edge_attr_rbf_l = pamnet.mlp_rbf_l(rbf_l)
        edge_attr_rbf_g = pamnet.mlp_rbf_g(rbf_g)
        edge_attr_sbf1 = pamnet.mlp_sbf1(sbf1)
        edge_attr_sbf2 = pamnet.mlp_sbf2(sbf2)

        # Message passing to collect node features at each layer
        out_global = []
        out_local = []
        att_score_global = []
        att_score_local = []
        node_features_per_layer = [x.clone()]
        
        for layer in range(pamnet.n_layer):
            # Global layer update
            x, out_g, att_score_g = pamnet.global_layer[layer](x, edge_attr_rbf_g, edge_index_g)
            out_global.append(out_g)
            att_score_global.append(att_score_g)

            # Local layer update
            x, out_l, att_score_l = pamnet.local_layer[layer](x, edge_attr_rbf_l, edge_attr_sbf2, edge_attr_sbf1,
                                                            idx_kj, idx_ji, idx_jj_pair, idx_ji_pair, edge_index_l)
            out_local.append(out_l)
            att_score_local.append(att_score_l)
            
            # Save intermediate node representations for layer-wise distillation
            node_features_per_layer.append(x.clone())
        
        # Get node features from specified layer. Defaults - final layer (-1).
        if layer_idx == -1:
            node_feat = node_features_per_layer[-1]
        else:
            layer_idx = min(layer_idx, len(node_features_per_layer) - 1)
            node_feat = node_features_per_layer[layer_idx]
        
        # Fusion module 
        att_score = torch.cat((torch.cat(att_score_global, 0), torch.cat(att_score_local, 0)), -1)
        att_score = F.leaky_relu(att_score, 0.2)
        att_weight = pamnet.softmax(att_score)

        out = torch.cat((torch.cat(out_global, 0), torch.cat(out_local, 0)), -1)
        out = (out * att_weight).sum(dim=-1)
        out = out.sum(dim=0).unsqueeze(-1)
        out = global_add_pool(out, batch)
        
        # Derive graph-level features via pooling
        graph_feat = global_mean_pool(node_feat, batch)
        
        return out.view(-1), node_feat, graph_feat


# Early stopper 
class EarlyStopper:
    def __init__(self, patience=20, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.best_state_dict = None
        
    def __call__(self, val_loss, model, epoch):
        # Validation loss improved beyond the minimum required delta
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            self.counter = 0
            return False
        
        # Validation loss did not improve
        self.counter += 1
        return self.counter >= self.patience
    
    def restore_best_model(self, model):
        if self.best_state_dict:
            model.load_state_dict(self.best_state_dict)
            print(f"[INFO] Restored best model weights from epoch {self.best_epoch + 1}")



# Feature aligner: reconciles dimensional or topological differences between Teacher and Student
class FeatureAligner(nn.Module):
    def __init__(self, student_dim, teacher_dim, hidden_dim=None, pool_type='mean'):
        super().__init__()
        self.student_dim = student_dim
        self.teacher_dim = teacher_dim
        self.pool_type = pool_type
        
        # Create a projection layer if hidden dimensions do not match
        if student_dim != teacher_dim:
            if hidden_dim is not None:
                self.proj = nn.Sequential(
                    nn.Linear(student_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, teacher_dim)
                )
            else:
                self.proj = nn.Linear(student_dim, teacher_dim)
        else:
            self.proj = nn.Identity()
    
    def pool_nodes_to_graph(self, node_feat, batch):
        if self.pool_type == 'mean':
            return global_mean_pool(node_feat, batch)
        else:
            return global_add_pool(node_feat, batch)
    
    def forward(self, feat_s, feat_t, batch=None):
        # Apply pooling if batch dimensions mismatch
        if feat_s.shape[0] != feat_t.shape[0] and batch is not None:
            if feat_s.shape[0] > feat_t.shape[0]:
                # Student is node-level; pool to graph-level
                feat_s = self.pool_nodes_to_graph(feat_s, batch)
            else:
                # Teacher is node-level; pool to graph-level
                feat_t = self.pool_nodes_to_graph(feat_t, batch)
        
        # Feature projection
        feat_s = self.proj(feat_s)       
        return feat_s, feat_t


# Setup_environment & normalization
def setup_device(gpu_id=0):
    if torch.cuda.is_available():
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(gpu_id)}")
        return torch.device(f"cuda:{gpu_id}")
    print("[INFO] Using CPU")
    return torch.device("cpu")

def set_seed(seed):
    # Fix all random seeds for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def compute_norm_params(y_data):
    max_val, min_val = float(np.max(y_data)), float(np.min(y_data))
    print(f"[INFO] MinMax stats: min={min_val:.8f}, max={max_val:.8f}, range={max_val-min_val:.8f}")
    return {'max_val': max_val, 'min_val': min_val}

def normalize(y, norm_params):
    denom = norm_params['max_val'] - norm_params['min_val']
    return (y - norm_params['min_val']) / denom if abs(denom) > 1e-8 else y * 0

def denormalize(y, norm_params):
    return y * (norm_params['max_val'] - norm_params['min_val']) + norm_params['min_val']


# Teacher loading
def load_teacher(checkpoint_path, model_type, dim, n_layer, cutoff_l, cutoff_g, device):
    print(f"[INFO] Loading Teacher checkpoint from: {checkpoint_path}")
    print(f"[INFO] Teacher configuration: model={model_type}, dim={dim}, n_layer={n_layer}, cutoff=({cutoff_l}, {cutoff_g})")
    
    config = Config(dim=dim, n_layer=n_layer, cutoff_l=cutoff_l, cutoff_g=cutoff_g)
    teacher = PAMNet(config) if model_type == 'pamnet' else MXMNet(config)
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    
    # Fix dict keys if the output head names changed across versions
    fixed_dict = {k.replace("output_head.0.0.", "output_head."): v for k, v in state_dict.items()}
    teacher.load_state_dict(fixed_dict, strict=False)
    teacher = teacher.to(device)
    teacher.eval() # Always lock the teacher's weights
    
    # Wrapping PAMNet with feature extraction support
    if model_type == 'pamnet':
        teacher = PAMNetWithFeatures(teacher)
    
    return teacher

def get_teacher_output(teacher, batch, distill_layer=-1):
    # Retrieves both the prediction and the intermediate representations
    if hasattr(teacher, 'forward_with_features'):
        return teacher.forward_with_features(batch, layer_idx=distill_layer) if distill_layer != -1 else teacher.forward_with_features(batch)
    return teacher(batch), None, None


# Feature shape analysis
def analyze_feature_shapes(student, teacher, sample_batch, device, args):
    print("\n[INFO] Analyzing feature shapes for representation distillation...")   
    sample_batch = sample_batch.to(device)
    
    with torch.no_grad():
        y_t, feat_t, _ = get_teacher_output(teacher, sample_batch, args.distill_layer)
        student.eval()
        y_s, feat_s, _ = student.forward_with_features(sample_batch)
    
    batch_size = sample_batch.y.shape[0] if hasattr(sample_batch, 'y') else sample_batch.num_graphs
    num_nodes = sample_batch.x.shape[0] if hasattr(sample_batch, 'x') else sample_batch.z.shape[0]

    # Evaluate teacher shape output
    if feat_t is not None:
        print(f"[INFO] Teacher feature shape: {feat_t.shape}")
        t_is_node = feat_t.shape[0] == num_nodes
        t_is_graph = feat_t.shape[0] == batch_size
        t_dim = feat_t.shape[-1]
        print(f"[INFO] Teacher hierarchy: {'node-level' if t_is_node else 'graph-level' if t_is_graph else 'unknown'}, dimension={t_dim}")
    else:
        print(f"[INFO] Teacher feature: None")
        t_is_node, t_is_graph, t_dim = False, False, None
    
    # Evaluate student shape output
    if feat_s is not None:
        print(f"[INFO] Student feature shape: {feat_s.shape}")
        s_is_node = feat_s.shape[0] == num_nodes
        s_is_graph = feat_s.shape[0] == batch_size
        s_dim = feat_s.shape[-1]
        print(f"[INFO] Student hierarchy: {'node-level' if s_is_node else 'graph-level' if s_is_graph else 'unknown'}, dimension={s_dim}")
    else:
        print(f"[INFO] Student feature: None")
        s_is_node, s_is_graph, s_dim = False, False, None
    
    # Evaluate compatibility and determine alignment strategy
    can_align = False
    align_strategy = None
    
    if feat_t is None or feat_s is None:
        print(f"[WARNING] Feature distillation NOT possible: missing features in one or both models.")
        align_strategy = 'disabled'
    elif feat_t.shape == feat_s.shape:
        print(f"[INFO] Features already aligned perfectly! No structural projection needed.")
        can_align = True
        align_strategy = 'none'
    elif t_is_node and s_is_node and t_dim != s_dim:
        print(f"[INFO] Both are node-level, but dimensions mismatch. Will inject linear projection layer.")
        can_align = True
        align_strategy = 'project_dim_node'
    elif t_is_graph and s_is_graph and t_dim != s_dim:
        print(f"[INFO] Both are graph-level, but dimensions mismatch. Will inject linear projection layer.")
        can_align = True
        align_strategy = 'project_dim_graph'
    elif t_is_node and s_is_graph:
        print(f"[INFO] Topological mismatch: Teacher=node, Student=graph. Will apply pooling to teacher features.")
        can_align = True
        align_strategy = 'pool_teacher'
    elif t_is_graph and s_is_node:
        print(f"[INFO] Topological mismatch: Teacher=graph, Student=node. Will apply pooling to student features.")
        can_align = True
        align_strategy = 'pool_student'
    else:
        print(f"[WARNING] Cannot automatically align features using known strategies!")
        align_strategy = 'incompatible'
    
    return {
        'can_align': can_align,
        'strategy': align_strategy,
        'teacher_shape': feat_t.shape if feat_t is not None else None,
        'student_shape': feat_s.shape if feat_s is not None else None,
        'teacher_dim': t_dim,
        'student_dim': s_dim,
        't_is_node': t_is_node,
        's_is_node': s_is_node,
    }

    
# Stage 1: Knowledge Distillation
def train_distillation(student, teacher, train_loader, val_loader, device, args, feat_info=None):
    print("\n" + "="*70)
    print("STAGE 1: KNOWLEDGE DISTILLATION")
    print("="*70)
    
    # Feature aligner setup
    feat_aligner = None
    use_feature_loss = args.feature_loss_weight > 0
    
    if use_feature_loss and feat_info is not None:
        if feat_info['can_align'] and feat_info['strategy'] != 'disabled':
            print(f"[INFO] Activating alignment strategy: {feat_info['strategy']}")
            
            student_dim = feat_info['student_dim'] or args.hidden_dim
            teacher_dim = feat_info['teacher_dim'] or args.teacher_dim
            
            feat_aligner = FeatureAligner(
                student_dim=student_dim,
                teacher_dim=teacher_dim,
                hidden_dim=args.feat_proj_hidden,
                pool_type='mean'
            ).to(device)
            
            print(f"[INFO] Aligner successfully mapped dimensions: {student_dim} -> {teacher_dim}")
        else:
            print(f"[WARNING] Feature distillation DISABLED due to strategy: {feat_info['strategy']}")
            use_feature_loss = False
    elif use_feature_loss:
        print(f"[WARNING] No feature info available, attempting to process without aligner.")
    else:
        print(f"[INFO] Feature loss fully disabled (weight={args.feature_loss_weight})")
    
    # Configure optimizer and add feature aligner parameters if it exists
    params_to_optimize = list(student.parameters())
    if feat_aligner is not None:
        params_to_optimize += list(feat_aligner.parameters())
    
    optimizer = Adam(params_to_optimize, lr=args.distill_lr, weight_decay=args.wd)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)
    es = EarlyStopper(args.es_patience, args.es_min_delta) if args.early_stopping else None
    
    feature_loss_count = 0
    feature_loss_skip_count = 0
    
    logs = []
    for epoch in range(args.distill_epochs):
        student.train()
        if feat_aligner:
            feat_aligner.train()
        
        total_loss = 0
        total_output_loss = 0
        total_feature_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Distill E{epoch+1}/{args.distill_epochs}", leave=False):
            batch = batch.to(device)
            
            # Extract frozen teacher embeddings and predictions
            with torch.no_grad():
                y_t, feat_t, _ = get_teacher_output(teacher, batch, args.distill_layer)
            
            # Forward pass through the student model
            y_s, feat_s, _ = student.forward_with_features(batch)
            
            # Calculate task-specific distillation loss
            output_loss = F.mse_loss(y_s.squeeze(), y_t.squeeze())
            loss = output_loss
            total_output_loss += output_loss.item()
            
            # Feature loss
            if use_feature_loss and feat_t is not None and feat_s is not None:
                try:
                    batch_idx = batch.batch if hasattr(batch, 'batch') else None
                    
                    if feat_aligner is not None:
                        feat_s_aligned, feat_t_aligned = feat_aligner(feat_s, feat_t, batch_idx)
                    else:
                        feat_s_aligned, feat_t_aligned = feat_s, feat_t
                    
                    # Compute feature discrepancy if shapes match
                    if feat_s_aligned.shape == feat_t_aligned.shape:
                        feature_loss = args.feature_loss_weight * F.mse_loss(feat_s_aligned, feat_t_aligned)
                        loss = loss + feature_loss
                        total_feature_loss += feature_loss.item()
                        feature_loss_count += 1
                    else:
                        feature_loss_skip_count += 1
                except Exception as e:
                    if feature_loss_skip_count == 0:
                        print(f"\n[WARNING] Feature loss encountered an error during calculation: {e}")
                    feature_loss_skip_count += 1
            
            # Backpropagation and optimization step
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_to_optimize, 1.0) # Prevent exploding gradients
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        student.eval()
        if feat_aligner:
            feat_aligner.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                y_t, _, _ = get_teacher_output(teacher, batch)
                y_s, _, _ = student.forward_with_features(batch)
                val_loss += F.mse_loss(y_s.squeeze(), y_t.squeeze()).item()
        
        n_batches = len(train_loader)
        avg_train = total_loss / n_batches
        avg_val = val_loss / len(val_loader)
        avg_output = total_output_loss / n_batches
        avg_feature = total_feature_loss / n_batches if feature_loss_count > 0 else 0
        
        scheduler.step(avg_val)
        logs.append({
            'epoch': epoch+1, 
            'train_loss': avg_train, 
            'val_loss': avg_val,
            'output_loss': avg_output,
            'feature_loss': avg_feature
        })
        
        if (epoch + 1) % 10 == 0:
            feat_str = f" | Feat: {avg_feature:.6f}" if feature_loss_count > 0 else ""
            print(f"E{epoch+1:03d} | Train MSE: {avg_train:.6f} | Val MSE: {avg_val:.6f} | Out: {avg_output:.6f}{feat_str}")
        
        if es and es(avg_val, student, epoch):
            print(f"[Early Stop] Distillation halted early at epoch {epoch+1} due to no validation improvement.")
            es.restore_best_model(student)
            break
    
    if use_feature_loss:
        if feature_loss_count > 0:
            print(f"[INFO] Representation distillation completed successfully.")
            print(f"[INFO] Processed {feature_loss_count} batches using feature constraints.")
            if feature_loss_skip_count > 0:
                print(f"[WARNING] Skipped feature constraints for {feature_loss_skip_count} batches due to anomalies.")
        else:
            print(f"[WARNING] Feature distillation was INACTIVE because no valid mappings were found.")
    
    return logs


# Stage 2: Fine-Tuning
def train_finetune(student, train_loader, val_loader, device, args, norm_params):
    print("\n" + "="*70)
    print("STAGE 2: FINE-TUNING ON TRUE LABELS")
    print("="*70)
    
    optimizer = Adam(student.parameters(), lr=args.ft_lr, weight_decay=args.wd)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)
    es = EarlyStopper(args.es_patience, args.es_min_delta) if args.early_stopping else None
    
    logs = []
    for epoch in range(args.ft_epochs):
        student.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"FT E{epoch+1}/{args.ft_epochs}", leave=False):
            batch = batch.to(device)
            # Normalize ground truth labels according to training distribution
            y_true_norm = normalize(batch.y.squeeze(), norm_params)
            
            y_s, _, _ = student.forward_with_features(batch)
            loss = F.l1_loss(y_s.squeeze(), y_true_norm)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        student.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                y_true_norm = normalize(batch.y.squeeze(), norm_params)
                y_s, _, _ = student.forward_with_features(batch)
                val_loss += F.l1_loss(y_s.squeeze(), y_true_norm).item()
        
        avg_train, avg_val = total_loss / len(train_loader), val_loss / len(val_loader)
        scheduler.step(avg_val)
        logs.append({'epoch': epoch+1, 'train_loss': avg_train, 'val_loss': avg_val})
        
        if (epoch + 1) % 10 == 0:
            print(f"E{epoch+1:03d} | Train MAE: {avg_train:.6f} | Val MAE: {avg_val:.6f}")
        
        if es and es(avg_val, student, epoch):
            print(f"[Early Stop] Fine-tuning halted early at epoch {epoch+1}")
            es.restore_best_model(student)
            break
    
    return logs


# Input dimension detection
def detect_student_in_dim(args, full_dataset):
    # Determines the correct input dimension size by inspecting the dataset format
    sample = full_dataset[0]
    
    if args.student_in_dim is not None:
        print(f"[INFO] Bypassing auto-detection, using user-specified input dimension: {args.student_in_dim}")
        return args.student_in_dim
    
    if hasattr(sample, 'x') and sample.x is not None:
        if sample.x.dim() == 1:
            # Format indicates 1D atomic indices (e.g., categorical atomic numbers)
            n_samples = min(100, len(full_dataset))
            max_z = max(full_dataset[i].x.max().item() for i in range(n_samples))
            in_dim = int(max_z) + 1
            print(f"[INFO] Auto-detected atomic indices format. Maximum atomic index found={max_z}. Setting input dim={in_dim}")
            return in_dim
        else:
            # Format indicates embedded or multi-dimensional node features
            in_dim = sample.x.shape[-1]
            print(f"[INFO] Auto-detected multi-dimensional node features. Setting input dim={in_dim}")
            return in_dim
    
    if hasattr(sample, 'z') and sample.z is not None:
        # Standard molecular graph format using nuclear charges ('z')
        n_samples = min(100, len(full_dataset))
        max_z = max(full_dataset[i].z.max().item() for i in range(n_samples))
        in_dim = int(max_z) + 1
        print(f"[INFO] Auto-detected atomic numbers ('z'). Maximum Z={max_z}. Setting input dim={in_dim}")
        return in_dim
    
    print(f"[INFO] Unable to automatically detect input dimensions. Falling back to default hidden_dim: {args.hidden_dim}")
    return args.hidden_dim


# Train mode
def train_mode(args, device):
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.run_label}" if args.run_label else ""
    run_name = f"{args.student_model}_{args.dataset}_{args.n_body}body_h{args.hidden_dim}_l{args.num_layers}{suffix}_{timestamp}"
    
    labels_to_load = [int(l) for l in args.labels.split(',')] if args.labels else None
    
    print("\n" + "="*70)
    print("PES TRAINING (Train Mode)")
    print("="*70)
    print(f"Run Profile: {run_name}")
    print(f"Dataset Target: {args.dataset}, System: {args.n_body}-body, Target Labels: {labels_to_load}")
    print(f"Teacher Anchor: {args.teacher_model} (dim={args.teacher_dim}, layers={args.teacher_n_layer})")
    print(f"Student Architecture: {args.student_model} (hidden={args.hidden_dim}, layers={args.num_layers})")
    print(f"Distillation Configuration: {args.distill_epochs} epochs (λ={args.feature_loss_weight}, anchor layer={args.distill_layer})")
    print(f"Fine-tune Calibration: {args.ft_epochs} epochs")
    print(f"Batch Configuration: {args.batch_size}, Split Policy: Train=90% / Val=10%")
    print("="*70)
    
    # Load teacher
    teacher = load_teacher(
        args.teacher_checkpoint, args.teacher_model,
        args.teacher_dim, args.teacher_n_layer,
        args.teacher_cutoff_l, args.teacher_cutoff_g, device
    )
    
    # Load data
    data_path = args.data_path or f'../dataset/{args.dataset}/{args.dataset}_{args.n_body}body_energy_application.npz'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data package not found at: {data_path}")
    
    print(f"\n[INFO] Loading dataset from: {data_path}")
    full_dataset = ApplicationDS(
        root=f'pes_{args.dataset}_{args.n_body}body{suffix}',
        src=data_path,
        labels=labels_to_load,
        mean_val=None, std_val=None,
        pre_transform=Distance(norm=False)
    )
    
    total = len(full_dataset)
    train_size = int(total * 0.9)
    val_size = total - train_size
    print(f"[INFO] Corpus metrics: Total={total}, Train Partition={train_size} (90%), Validation Partition={val_size} (10%)")
    
    gen = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size], generator=gen)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    
    print("\n[INFO] Computing normalization bounds for training partition.")
    train_ys = np.array([full_dataset[i].y.item() for i in train_ds.indices])
    norm_params = compute_norm_params(train_ys)
    
    # Initialize student
    sample = full_dataset[0]
    edge_dim = sample.edge_attr.shape[1] if hasattr(sample, 'edge_attr') and sample.edge_attr is not None else None
    student_in_dim = detect_student_in_dim(args, full_dataset)
    
    student = get_student_model(
        name=args.student_model,
        in_dim=student_in_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        edge_dim=edge_dim
    ).to(device)
    
    n_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"[INFO] Student initialized successfully: {args.student_model}")
    print(f"[INFO] Student tensor shapes: in_dim={student_in_dim}, hidden_dim={args.hidden_dim}, "
          f"num_layers={args.num_layers}, edge_dim={edge_dim}")
    print(f"[INFO] Total trainable parameters: {n_params:,}")
    
    # Analyze feature shapes
    feat_info = None
    if args.feature_loss_weight > 0:
        sample_batch = next(iter(train_loader))
        feat_info = analyze_feature_shapes(student, teacher, sample_batch, device, args)
    
    # Executio
    start_time = time.time()
    all_logs = {}
    
    # Distillation
    distill_start = time.time()
    all_logs['distillation'] = train_distillation(student, teacher, train_loader, val_loader, device, args, feat_info)
    distill_time = time.time() - distill_start
    
    # Fine-tuning
    ft_start = time.time()
    all_logs['finetune'] = train_finetune(student, train_loader, val_loader, device, args, norm_params)
    ft_time = time.time() - ft_start
    
    total_time = time.time() - start_time
    
    # Evaluation
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    
    student.eval()
    
    def eval_loader(loader, norm_params):
        y_pred_list, y_true_list = [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                pred, _, _ = student.forward_with_features(batch)
                y_pred_list.append(pred.cpu().numpy())
                y_true_list.append(batch.y.cpu().numpy())
                
        y_pred_norm = np.concatenate(y_pred_list).flatten()
        y_true = np.concatenate(y_true_list).flatten()
        
        # Invert normalization to compute raw metrics
        y_pred = denormalize(y_pred_norm, norm_params)
        
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'mae_kcal': mean_absolute_error(y_true, y_pred) * HARTREE_TO_KCAL_MOL,
            'r2': r2_score(y_true, y_pred)
        }
    
    train_metrics = eval_loader(train_loader, norm_params)
    val_metrics = eval_loader(val_loader, norm_params)
    
    print(f"\n{'Split':<6} | {'MAE(Ha)':>12} | {'MAE(kcal/mol)':>14} | {'R²':>8}")
    print("-" * 50)
    print(f"{'train':<6} | {train_metrics['mae']:>12.8f} | {train_metrics['mae_kcal']:>14.4f} | {train_metrics['r2']:>8.4f}")
    print(f"{'val':<6} | {val_metrics['mae']:>12.8f} | {val_metrics['mae_kcal']:>14.4f} | {val_metrics['r2']:>8.4f}")
    
    print(f"\nExecution Duration: Distill={distill_time/60:.1f}min, FT={ft_time/60:.1f}min, Total={total_time/60:.1f}min")
    
    print("\n" + "="*70)
    print("SAVING TO: " + args.save_dir)
    print("="*70)
    
    model_path = os.path.join(args.save_dir, f"model_{run_name}.pt")
    torch.save(student.state_dict(), model_path)
    print(f"[SAVED] {model_path}")
    
    norm_path = os.path.join(args.save_dir, f"norm_stats_{run_name}.npy")
    np.save(norm_path, norm_params)
    print(f"[SAVED] {norm_path}")
    
    results = {
        'metrics': {'train': train_metrics, 'val': val_metrics},
        'norm_params': norm_params,
        'args': vars(args),
        'student_in_dim': student_in_dim,
        'feature_info': feat_info,
        'time': {'distill': distill_time, 'finetune': ft_time, 'total': total_time},
        'data_sizes': {'train': train_size, 'val': val_size},
        'timestamp': timestamp,
        'n_params': n_params  
    }
    results_path = os.path.join(args.save_dir, f"results_{run_name}.npy")
    np.save(results_path, results)
    print(f"[SAVED] {results_path}")
    
    logs_path = os.path.join(args.save_dir, f"logs_{run_name}.json")
    with open(logs_path, 'w') as f:
        json.dump(all_logs, f, indent=2)
    print(f"[SAVED] {logs_path}")
    

# Prediction mode for evaluating trained models on new datasets
def predict_mode(args, device):
    print("\n" + "="*70)
    print("PREDICTION INFERENCE (Predict Mode)")
    print("="*70)
    
    # Input boundary checks
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Missing Model Weights Artifact: {args.model_path}")
    if not os.path.exists(args.norm_stats_path):
        raise FileNotFoundError(f"Missing Normalization Stats Artifact: {args.norm_stats_path}")
    if not os.path.exists(args.predict_data_path):
        raise FileNotFoundError(f"Missing Evaluation Payload: {args.predict_data_path}")
    
    print(f"Referenced Model Weights: {args.model_path}")
    print(f"Referenced Norm Profile:  {args.norm_stats_path}")
    print(f"Target Data Payload:      {args.predict_data_path}")
    print("="*70)
    
    # Load normalization parameters (from training)
    norm_params = np.load(args.norm_stats_path, allow_pickle=True).item()
    print(f"\n[INFO] Loaded Normalization Scales (from training snapshot):")
    print(f"  Maximum Bound: {norm_params['max_val']:.8f} Ha")
    print(f"  Minimum Bound: {norm_params['min_val']:.8f} Ha")
    
    # Load prediction dataset (same format as training)
    dataset_name = os.path.basename(args.predict_data_path).replace('.npz', '')
    labels_to_load = [int(l) for l in args.labels.split(',')] if args.labels else None
    
    print(f"\n[INFO] Injecting prediction corpus: {args.predict_data_path}")
    print(f"  Identified Namespace: {dataset_name}")
    print(f"  Requested Label IDs: {labels_to_load}")
    
    dataset = ApplicationDS(
        root=f'predict_{dataset_name}',
        src=args.predict_data_path,
        labels=labels_to_load,
        mean_val=None,
        std_val=None,
        pre_transform=Distance(norm=False)
    )
    
    print(f"[INFO] Loaded {len(dataset)} samples")
    
    # Create data loader
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Detect input dimension from data
    sample = dataset[0]
    edge_dim = sample.edge_attr.shape[1] if hasattr(sample, 'edge_attr') and sample.edge_attr is not None else None
    student_in_dim = detect_student_in_dim(args, dataset)
    
    # Create and load student model
    print(f"\n[INFO] Creating student model:")
    print(f"  Model:      {args.student_model}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Num layers: {args.num_layers}")
    print(f"  Input dim:  {student_in_dim}")
    print(f"  Edge dim:   {edge_dim}")
    
    student = get_student_model(
        name=args.student_model,
        in_dim=student_in_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        edge_dim=edge_dim
    ).to(device)
    
    # Load trained weights
    student.load_state_dict(torch.load(args.model_path, map_location=device))
    student.eval()
    
    n_params = sum(p.numel() for p in student.parameters())
    print(f"[INFO] System locked and loaded successfully.")
    print(f"  Capacity: {n_params:,} parameters ({n_params/1000:.1f}K)")
    
    # Run prediction
    print("\n" + "="*70)
    print("RUNNING PREDICTION")
    print("="*70)
    
    start_time = time.time()
    y_pred_list = []
    y_true_list = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference Processing"):
            batch = batch.to(device)
            
            # Get prediction
            pred, _, _ = student.forward_with_features(batch)
            
            y_pred_list.append(pred.cpu().numpy())
            y_true_list.append(batch.y.cpu().numpy())
    
    # Denormalize predictions
    y_pred_norm = np.concatenate(y_pred_list).flatten()
    y_true = np.concatenate(y_true_list).flatten()
    y_pred = denormalize(y_pred_norm, norm_params)
    
    # Compute metrics
    mae_ha = mean_absolute_error(y_true, y_pred)
    mae_kcal = mae_ha * HARTREE_TO_KCAL_MOL
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    total_time = time.time() - start_time
    
    print(f"\nDataset Info:")
    print(f"  Samples:     {len(y_true)}")
    print(f"  Time:        {total_time:.2f} sec ({total_time/60:.1f} min)")
    print(f"  Speed:       {len(y_true)/total_time:.1f} samples/sec")
    
    # Prediction statistics
    print(f"\nPrediction Statistics:")
    print(f"  True  - Mean: {np.mean(y_true):.8f}, Std: {np.std(y_true):.8f}")
    print(f"  Pred  - Mean: {np.mean(y_pred):.8f}, Std: {np.std(y_pred):.8f}")
    print(f"  Error - Mean: {np.mean(y_pred - y_true):.8f}, Std: {np.std(y_pred - y_true):.8f}")
    
    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    pred_data = {
        'y_pred': y_pred,
        'y_true': y_true,
        'metrics': {
            'mae': mae_ha,
            'mae_kcal': mae_kcal,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        },
        'dataset': dataset_name,
        'model_path': args.model_path,
        'norm_stats_path': args.norm_stats_path,
        'data_path': args.predict_data_path,
        'model_config': {
            'model': args.student_model,
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'n_params': n_params
        },
        'timestamp': timestamp
    }
    
    save_path = os.path.join(args.save_dir, f"predictions_{args.student_model}_{dataset_name}_{timestamp}.npy")
    np.save(save_path, pred_data)
    print(f"[SAVED ARTIFACT] {save_path}")
    
    print("\n" + "="*70)
    print("PREDICTION COMPLETE")
    print("="*70)
    print(f"Dataset: {dataset_name}")
    print(f"Model:   {args.student_model} (h={args.hidden_dim}, l={args.num_layers})")
    print(f"MAE:     {mae_kcal:.4f} kcal/mol")
    print(f"R²:      {r2:.4f}")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="PES Training & Prediction Environment")   
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict'])
    
    # Data 
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--n_body', type=int, choices=[2, 3])
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--labels', type=str, default=None)
    
    # Teacher 
    parser.add_argument('--teacher_checkpoint', type=str)
    parser.add_argument('--teacher_model', type=str, default='pamnet', choices=['pamnet', 'mxmnet'])
    parser.add_argument('--teacher_dim', type=int, default=128)
    parser.add_argument('--teacher_n_layer', type=int, default=5)
    parser.add_argument('--teacher_cutoff_l', type=float, default=5.0)
    parser.add_argument('--teacher_cutoff_g', type=float, default=10.0)
 
    # Student 
    parser.add_argument('--student_model', type=str, required=True)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--student_in_dim', type=int, default=None)
    
    # Distillation 
    parser.add_argument('--distill_epochs', type=int, default=150)
    parser.add_argument('--distill_lr', type=float, default=1e-4)
    parser.add_argument('--feature_loss_weight', type=float, default=0.01)
    parser.add_argument('--distill_layer', type=int, default=-1)
    parser.add_argument('--feat_proj_hidden', type=int, default=None)
    
    # Fine-tuning 
    parser.add_argument('--ft_epochs', type=int, default=200)
    parser.add_argument('--ft_lr', type=float, default=1e-5)
    
    # Train mode
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--wd', type=float, default=1e-5)
    parser.add_argument('--early_stopping', action='store_true')
    parser.add_argument('--es_patience', type=int, default=20)
    parser.add_argument('--es_min_delta', type=float, default=1e-6)
    
    # Predict mode
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--norm_stats_path', type=str)
    parser.add_argument('--predict_data_path', type=str)
    
    # System
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='./pes_results')
    parser.add_argument('--run_label', type=str, default='')
    
    args = parser.parse_args()
    device = setup_device(args.gpu)
    
    if args.mode == 'train':
        if not all([args.dataset, args.n_body, args.teacher_checkpoint]):
            raise ValueError("Training execution blocked: Requires --dataset, --n_body, and --teacher_checkpoint flags.")
        train_mode(args, device)
    else:
        if not all([args.model_path, args.norm_stats_path, args.predict_data_path]):
            raise ValueError("Inference execution blocked: Requires --model_path, --norm_stats_path, and --predict_data_path flags.")
        predict_mode(args, device)


if __name__ == "__main__":
    main()
