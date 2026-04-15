from __future__ import division, print_function

import argparse
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch_geometric.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
import os
import sys
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from utils import EMA
from model import MXMNet, Config
from mixed_body_dataset import MixedBodyDS
from pamnet import PAMNet
from tool_utils import load_daset_config, load_daset_config_water

HA_TO_KCAL = 627.5094740631

# Preserves normalization parameters when filtering
class FilteredDataset(Dataset):
    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = indices
        self.max_val = base_dataset.max_val
        self.min_val = base_dataset.min_val
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.base_dataset[self.indices[idx]]


# Manages top-k best checkpoints
class CheckpointManager:
    def __init__(self, save_dir, k=3):
        self.save_dir = save_dir
        self.k = k
        self.checkpoints = []
        os.makedirs(self.save_dir, exist_ok=True)

    def save(self, state_dict, val_loss, filename_suffix):
        ckpt_path = os.path.join(self.save_dir, filename_suffix)
        should_save = len(self.checkpoints) < self.k or val_loss < self.checkpoints[-1]['loss']
        
        if should_save:
            torch.save(state_dict, ckpt_path)
            self.checkpoints.append({'loss': val_loss, 'path': ckpt_path})
            self.checkpoints.sort(key=lambda x: x['loss'])
            
            if len(self.checkpoints) > self.k:
                to_remove = self.checkpoints.pop()
                if os.path.exists(to_remove['path']):
                    try:
                        os.remove(to_remove['path'])
                    except OSError:
                        pass
            
            print(f"[Ckpt] Saved: {filename_suffix} (Val: {val_loss:.6f})")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def denormalize(y, max_val, min_val):
    if max_val is None or min_val is None:
        return y
    return y * (max_val - min_val) + min_val


def get_model(args, device):
    config = Config(dim=args.dim, n_layer=args.n_layer, 
                   cutoff_l=args.cutoff_l, cutoff_g=args.cutoff_g)
    model = PAMNet(config) if args.model == 'pamnet' else MXMNet(config)
    print(f'[Model] {args.model}: {args.n_layer} layers, dim={args.dim}, cutoffs=({args.cutoff_l}, {args.cutoff_g})')
    return model.to(device)


def test_model(loader, ema, model, device):
    ema.assign(model)
    model.eval()
    y_predict, y_true = [], []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            y_predict.append(output.detach().cpu().numpy())
            y_true.append(data.y.cpu().numpy())
    
    ema.resume(model)
    y_predict = denormalize(np.concatenate(y_predict), loader.dataset.max_val, loader.dataset.min_val)
    y_true = denormalize(np.concatenate(y_true), loader.dataset.max_val, loader.dataset.min_val)
    
    return {
        'mae': mean_absolute_error(y_true, y_predict),
        'r2': r2_score(y_true, y_predict)
    }


# Run prediction
def predict_model(loader, model, device, max_val, min_val):
    model.eval()
    y_predict, y_true = [], []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            y_predict.append(output.detach().cpu().numpy())
            y_true.append(data.y.cpu().numpy())
    
    y_predict = np.concatenate(y_predict)
    y_true = np.concatenate(y_true)
    
    # Denormalize
    y_predict = denormalize(y_predict, max_val, min_val)
    y_true = denormalize(y_true, max_val, min_val)
    
    return y_true.flatten(), y_predict.flatten()


def analyze_energy_distribution(dataset, dataset_name='dataset'):
    print(f"\n{'='*80}\nENERGY DISTRIBUTION: {dataset_name}\n{'='*80}")
    
    all_energies = []
    for i in range(len(dataset)):
        y_norm = dataset[i].y.item()
        y_real = y_norm * (dataset.max_val - dataset.min_val) + dataset.min_val
        all_energies.append(y_real)
    
    all_energies = np.array(all_energies)
    abs_energies = np.abs(all_energies)
    
    print(f"Percentiles (absolute energy):")
    for p in [25, 50, 75, 90, 95]:
        val = np.percentile(abs_energies, p)
        print(f"  {p:2d}%: {val:.6e} Ha")
    
    threshold_high = np.percentile(abs_energies, 75)
    threshold_med = np.percentile(abs_energies, 50)
    
    print(f"\nSuggested Thresholds:")
    print(f"  Stage 1: |E| > {threshold_high:.6e} Ha  (top 25%)")
    print(f"  Stage 2: |E| > {threshold_med:.6e} Ha   (top 50%)")
    print(f"  Stage 3: All samples")
    print('='*80 + '\n')
    
    return {'threshold_high': threshold_high, 'threshold_med': threshold_med}


def filter_by_energy(dataset, threshold, name='dataset'):
    print(f"[Filter] {name}: |E| > {threshold:.2e} Ha")
    
    energies = []
    for i in range(len(dataset)):
        y_norm = dataset[i].y.item()
        y_real = y_norm * (dataset.max_val - dataset.min_val) + dataset.min_val
        energies.append(y_real)
    
    energies = np.array(energies)
    indices = np.where(np.abs(energies) > threshold)[0]
    
    print(f"         Selected: {len(indices)}/{len(dataset)} ({100*len(indices)/len(dataset):.1f}%)")
    if len(indices) > 0:
        sel_e = energies[indices]
        print(f"         Range: [{sel_e.min():.6e}, {sel_e.max():.6e}] Ha\n")
    
    return FilteredDataset(dataset, indices.tolist())


def train_one_stage(stage_name, train_loader, val_loader, test_loader, model, optimizer, 
                   scheduler, scaler, ema, device, args, ckpt_manager, start_epoch, 
                   num_epochs, patience, monitor='r2'):
    """
    Train one stage with configurable monitoring metric.
    
    Args:
        monitor: 'r2' or 'mae' - which metric to use for early stopping and checkpoint saving
    """
    print(f"\n{'='*80}\nSTAGE: {stage_name}\n{'='*80}")
    monitor_str = 'MAE (lower=better)' if monitor == 'mae' else 'R² (higher=better)'
    print(f"Epochs: {start_epoch}-{start_epoch+num_epochs-1}, LR: {optimizer.param_groups[0]['lr']:.2e}, "
          f"Batch: {args.batch_size}, Samples: {len(train_loader.dataset)}, Patience: {patience}")
    print(f"Monitor: {monitor_str}\n")
    
    # Initialize best metrics based on monitor type
    if monitor == 'mae':
        best_metric = float('inf')  # lower is better
        is_better = lambda new, old: new < old
    else:  # r2
        best_metric = -float('inf')  # higher is better
        is_better = lambda new, old: new > old
    
    best_val_loss = float('inf')
    best_r2 = -float('inf')
    best_epoch = start_epoch
    best_ckpt = None
    patience_counter = 0
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
        # Training
        model.train()
        loss_all = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            with autocast():
                loss = F.l1_loss(model(data), data.y)
            
            loss_all += loss.item() * data.num_graphs
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            ema(model)
        
        train_loss = loss_all / len(train_loader.dataset)
        if scheduler:
            scheduler.step()
        
        # Validation
        val_result = test_model(val_loader, ema, model, device)
        val_loss = val_result['mae']
        val_r2 = val_result['r2']
        
        # Select current metric based on monitor type
        current_metric = val_loss if monitor == 'mae' else val_r2
        
        if is_better(current_metric, best_metric):
            best_metric = current_metric
            best_val_loss = val_loss
            best_r2 = val_r2
            best_epoch = epoch
            patience_counter = 0
            
            test_result = test_model(test_loader, ema, model, device)
            ckpt_name = f"stage_{stage_name}_ep{epoch}_r2{test_result['r2']:.4f}.pt"
            
            state_dict = {
                'epoch': epoch,
                'stage': stage_name,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'val_loss': val_loss,
                'test_r2': test_result['r2'],
                'test_mae': test_result['mae'],
                'monitor': monitor
            }
            
            ckpt_manager.save(state_dict, val_loss, ckpt_name)
            best_ckpt = os.path.join(ckpt_manager.save_dir, ckpt_name)
            
            print(f"Ep {epoch:03d}: Train={train_loss:.6f}, Val={val_loss:.6f}, "
                  f"Test R²={test_result['r2']:.4f}, MAE={test_result['mae']:.6f}")
        else:
            patience_counter += 1
            print(f"Ep {epoch:03d}: Train={train_loss:.6f}, Val={val_loss:.6f}, "
                  f"R²={val_r2:.4f} [Patience {patience_counter}/{patience}]")
            
            if patience_counter >= patience:
                print(f"\n[Early Stop] No improvement for {patience} epochs\n")
                break
    
    print(f"[Complete] Best epoch: {best_epoch}, Best R²: {best_r2:.4f}, Best MAE: {best_val_loss:.6f}\n")
    return best_ckpt, best_r2


def run_staged_training(args, device, dataset_config, labels):
    dataset_name = dataset_config['name']
    n_body = str(args.n_body)
    
    print(f"\n{'='*80}\nSTAGED TRAINING: {n_body}-BODY {dataset_name.upper()}\n{'='*80}")
    print(f"Model: {args.model}, Labels: {labels}\n")
    
    # Load data
    train_path = f"../dataset/{dataset_name}/{n_body}body_energy_train.npz"
    test_path = f"../dataset/{dataset_name}/{n_body}body_energy_test.npz"
    val_path = f"../dataset/{dataset_name}/{n_body}body_energy_val.npz"
    
    train_data = np.load(train_path, allow_pickle=True)
    label_key = 'labels' if 'labels' in train_data.files else 'label'
    
    # Normalization 
    print("[Normalization] Standard MinMax")
    train_y = np.concatenate([train_data['y'][train_data[label_key] == int(l)] for l in labels])
    max_val = float(np.max(train_y))
    min_val = float(np.min(train_y))
    print(f"  Range: [{min_val:.8f}, {max_val:.8f}] Ha\n")
    
    labels_str = '_'.join(sorted([str(l) for l in labels]))
    unique_key = f"{dataset_name}_{labels_str}_{n_body}_body_minmax"
    
    print(f"[Cache] Using unique key: {unique_key}")
    
    # Create datasets
    train_full = MixedBodyDS(f'train_{unique_key}', train_path, labels, max_val, min_val)
    test_full = MixedBodyDS(f'test_{unique_key}', test_path, labels, max_val, min_val)
    val_full = MixedBodyDS(f'val_{unique_key}', val_path, labels, max_val, min_val)
    
    print(f"[Data] Train: {len(train_full)}, Val: {len(val_full)}, Test: {len(test_full)}")
    
    # Analyze distribution
    stats = analyze_energy_distribution(train_full, f"{n_body}B Training")
    threshold_s1 = args.threshold_high if args.threshold_high else stats['threshold_high']
    threshold_s2 = args.threshold_med if args.threshold_med else stats['threshold_med']
    
    # Initialize model
    model = get_model(args, device)
    
    if args.resume and os.path.exists(args.resume):
        print(f"\n[Transfer] Loading: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        loaded = {k: v for k, v in ckpt['model_state_dict'].items() 
                 if k in model.state_dict() and v.shape == model.state_dict()[k].shape}
        model.load_state_dict(loaded, strict=False)
        print(f"           Loaded {len(loaded)}/{len(ckpt['model_state_dict'])} params\n")
    
    ckpt_manager = CheckpointManager(f"./ckpt/{dataset_name}_{args.model}_{n_body}b_staged", k=3)
    
    def seed_worker(worker_id):
        np.random.seed(torch.initial_seed() % 2**32)
        random.seed(torch.initial_seed() % 2**32)
    
    g = torch.Generator().manual_seed(args.seed)
    
    # STAGE 1: High-Energy
    print(f"{'='*80}\nPREPARING STAGE 1: High-Energy\n{'='*80}")
    train_s1 = filter_by_energy(train_full, threshold_s1, "Train")
    val_s1 = filter_by_energy(val_full, threshold_s1, "Val")
    test_s1 = filter_by_energy(test_full, threshold_s1, "Test")
    
    train_loader_s1 = DataLoader(train_s1, args.batch_size, shuffle=True, 
                                 worker_init_fn=seed_worker, generator=g, num_workers=4, pin_memory=True)
    val_loader_s1 = DataLoader(val_s1, args.batch_size, num_workers=4, pin_memory=True)
    test_loader_s1 = DataLoader(test_s1, args.batch_size, num_workers=4, pin_memory=True)
    
    optimizer_s1 = optim.Adam(model.parameters(), lr=args.lr_s1, weight_decay=0)
    scheduler_s1 = torch.optim.lr_scheduler.ExponentialLR(optimizer_s1, gamma=0.9961697)
    scaler = GradScaler()
    ema = EMA(model, decay=0.999)
    
    best_ckpt_s1, best_r2_s1 = train_one_stage(
        "1_HighEnergy", train_loader_s1, val_loader_s1, test_loader_s1, 
        model, optimizer_s1, scheduler_s1, scaler, ema, device, args, ckpt_manager,
        1, args.epochs_s1, args.patience
    )
    
    # STAGE 2: Medium-Energy
    print(f"{'='*80}\nPREPARING STAGE 2: Medium-Energy\n{'='*80}")
    if best_ckpt_s1 and os.path.exists(best_ckpt_s1):
        model.load_state_dict(torch.load(best_ckpt_s1, map_location=device)['model_state_dict'])
        print(f"[Loaded] {os.path.basename(best_ckpt_s1)}\n")
    
    train_s2 = filter_by_energy(train_full, threshold_s2, "Train")
    val_s2 = filter_by_energy(val_full, threshold_s2, "Val")
    test_s2 = filter_by_energy(test_full, threshold_s2, "Test")
    
    train_loader_s2 = DataLoader(train_s2, args.batch_size, shuffle=True,
                                 worker_init_fn=seed_worker, generator=g, num_workers=4, pin_memory=True)
    val_loader_s2 = DataLoader(val_s2, args.batch_size, num_workers=4, pin_memory=True)
    test_loader_s2 = DataLoader(test_s2, args.batch_size, num_workers=4, pin_memory=True)
    
    optimizer_s2 = optim.Adam(model.parameters(), lr=args.lr_s2, weight_decay=0)
    scheduler_s2 = torch.optim.lr_scheduler.ExponentialLR(optimizer_s2, gamma=0.9961697)
    ema = EMA(model, decay=0.999)
    
    best_ckpt_s2, best_r2_s2 = train_one_stage(
        "2_MediumEnergy", train_loader_s2, val_loader_s2, test_loader_s2,
        model, optimizer_s2, scheduler_s2, scaler, ema, device, args, ckpt_manager,
        args.epochs_s1 + 1, args.epochs_s2, args.patience
    )
    
    # STAGE 3: Full Dataset
    print(f"{'='*80}\nPREPARING STAGE 3: Full Dataset\n{'='*80}")
    if best_ckpt_s2 and os.path.exists(best_ckpt_s2):
        model.load_state_dict(torch.load(best_ckpt_s2, map_location=device)['model_state_dict'])
        print(f"[Loaded] {os.path.basename(best_ckpt_s2)}\n")
    
    print(f"[Using] All {len(train_full)} training samples\n")
    
    train_loader_s3 = DataLoader(train_full, args.batch_size, shuffle=True,
                                 worker_init_fn=seed_worker, generator=g, num_workers=4, pin_memory=True)
    val_loader_s3 = DataLoader(val_full, args.batch_size, num_workers=4, pin_memory=True)
    test_loader_s3 = DataLoader(test_full, args.batch_size, num_workers=4, pin_memory=True)
    
    optimizer_s3 = optim.Adam(model.parameters(), lr=args.lr_s3, weight_decay=0)
    scheduler_s3 = torch.optim.lr_scheduler.ExponentialLR(optimizer_s3, gamma=0.9961697)
    ema = EMA(model, decay=0.999)
    
    # Determine monitor metric for Stage 3
    stage3_monitor = 'mae' if (dataset_name == 'mix' or n_body == '3') else 'r2'
    if stage3_monitor == 'mae':
        print(f"[Stage 3] Using MAE monitor (dataset={dataset_name}, n_body={n_body})\n")
    
    best_ckpt_s3, best_r2_s3 = train_one_stage(
        "3_FullDataset", train_loader_s3, val_loader_s3, test_loader_s3,
        model, optimizer_s3, scheduler_s3, scaler, ema, device, args, ckpt_manager,
        args.epochs_s1 + args.epochs_s2 + 1, args.epochs_s3, args.patience,
        monitor=stage3_monitor
    )
    
    print(f"\n{'='*80}\nCOMPLETED\n{'='*80}")
    print(f"Stage 1 R²: {best_r2_s1:.4f}")
    print(f"Stage 2 R²: {best_r2_s2:.4f}")
    print(f"Stage 3 R²: {best_r2_s3:.4f}")
    print(f"Final: {best_ckpt_s3}\n{'='*80}\n")
    
    return best_ckpt_s3


# Predict mode: load checkpoint and generate predictions
def run_prediction(args, device):
    dataset = str(args.dataset).strip()
    n_body = str(args.n_body).strip()
    
    # Load config
    if dataset.startswith('H2O_mix') or dataset == 'mix':
        all_configs = load_daset_config_water()
    else:
        all_configs = load_daset_config()
    
    if dataset not in all_configs:
        raise ValueError(f'Invalid dataset: {dataset}')
    
    config = all_configs[dataset]
    dataset_name = dataset
    
    # Get labels
    if args.labels:
        labels = args.labels.split(',')
    else:
        labels = config.get('labels', {}).get(n_body, [])
        if not isinstance(labels, list):
            labels = [labels] if labels else []
    
    print(f"\n{'='*60}\nPREDICTION MODE\n{'='*60}")
    print(f"Dataset: {dataset}, N-body: {n_body}, Labels: {labels}")
    print(f"Model: {args.model}, Checkpoint: {args.checkpoint}")
    print('='*60)
    
    # Load data paths
    train_path = f"../dataset/{dataset_name}/{n_body}body_energy_train.npz"
    test_path = f"../dataset/{dataset_name}/{n_body}body_energy_test.npz"
    val_path = f"../dataset/{dataset_name}/{n_body}body_energy_val.npz"
    
    # Calculate normalization from training data
    train_data = np.load(train_path, allow_pickle=True)
    label_key = 'labels' if 'labels' in train_data.files else 'label'
    train_y = np.concatenate([train_data['y'][train_data[label_key] == int(l)] for l in labels])
    max_val = float(np.max(train_y))
    min_val = float(np.min(train_y))
    print(f"[Normalization] Range: [{min_val:.8f}, {max_val:.8f}] Ha")
    
    # Create unique key
    labels_str = '_'.join(sorted([str(l) for l in labels]))
    unique_key = f"{dataset_name}_{labels_str}_{n_body}_body_minmax"
    
    # Load datasets
    train_full = MixedBodyDS(f'train_{unique_key}', train_path, labels, max_val, min_val)
    test_full = MixedBodyDS(f'test_{unique_key}', test_path, labels, max_val, min_val)
    val_full = MixedBodyDS(f'val_{unique_key}', val_path, labels, max_val, min_val)
    
    print(f"[Data] Train: {len(train_full)}, Val: {len(val_full)}, Test: {len(test_full)}")
    
    # Create loaders
    train_loader = DataLoader(train_full, args.batch_size, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_full, args.batch_size, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_full, args.batch_size, num_workers=4, pin_memory=True)
    
    # Load model
    model = get_model(args, device)
    
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"[Loaded] Epoch: {checkpoint.get('epoch', 'N/A')}, Stage: {checkpoint.get('stage', 'N/A')}")
    print(f"         Val Loss: {checkpoint.get('val_loss', 'N/A'):.6f}, Test R²: {checkpoint.get('test_r2', 'N/A'):.4f}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Predict on all splits
    print("\n[Predicting.]")
    results = {}
        
    for split_name, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
        y_true, y_pred = predict_model(loader, model, device, max_val, min_val)
        
        r2 = r2_score(y_true, y_pred)
        
        y_true_kcal = y_true * HA_TO_KCAL
        y_pred_kcal = y_pred * HA_TO_KCAL
        
        mae_kcal = mean_absolute_error(y_true_kcal, y_pred_kcal)
        me_kcal = np.mean(y_pred_kcal - y_true_kcal)  
        
        print(f"  {split_name}: R²={r2:.4f}, MAE={mae_kcal:.4f} kcal/mol, ME={me_kcal:.4f} kcal/mol")
        
        results[split_name] = {
            'y_true': y_true,
            'y_pred': y_pred,
            'r2': r2,
            'mae_ha': mean_absolute_error(y_true, y_pred),
            'mae_kcal': mae_kcal,
            'me_kcal': me_kcal
        }
    
    output_prefix = args.output_prefix if args.output_prefix else f"{dataset}_{n_body}b_{args.model}"
    output_path = os.path.join(args.output_dir, f'{output_prefix}_predictions.npz')
             
    np.savez(output_path,
             train_y_true=results['train']['y_true'],
             train_y_pred=results['train']['y_pred'],
             train_r2=results['train']['r2'],
             train_mae_kcal=results['train']['mae_kcal'],
             train_me_kcal=results['train']['me_kcal'],
             val_y_true=results['val']['y_true'],
             val_y_pred=results['val']['y_pred'],
             val_r2=results['val']['r2'],
             val_mae_kcal=results['val']['mae_kcal'],
             val_me_kcal=results['val']['me_kcal'],
             test_y_true=results['test']['y_true'],
             test_y_pred=results['test']['y_pred'],
             test_r2=results['test']['r2'],
             test_mae_kcal=results['test']['mae_kcal'],
             test_me_kcal=results['test']['me_kcal'],
             dataset=dataset,
             n_body=n_body,
             model=args.model,
             checkpoint=args.checkpoint,
             max_val=max_val,
             min_val=min_val)
    
    print(f"\n[Saved] {output_path}")
    return output_path


def train_staged_energy(args, device):
    dataset = str(args.dataset).strip()
    n_body = str(args.n_body).strip()
    
    # Load config
    if dataset.startswith('H2O_mix') or dataset == 'mix':
        all_configs = load_daset_config_water()
    else:
        all_configs = load_daset_config()
    
    if dataset not in all_configs:
        raise ValueError(f'Invalid dataset: {dataset}')
    
    config = all_configs[dataset]
    config['name'] = dataset
    
    # Get labels
    if args.labels:
        labels = args.labels.split(',')
    else:
        labels = config.get('labels', {}).get(n_body, [])
        if not isinstance(labels, list):
            labels = [labels] if labels else []
    
    if not labels:
        raise ValueError(f"No labels for dataset={dataset}, n_body={n_body}")
    
    print(f"\n{'='*60}\nCONFIGURATION\n{'='*60}")
    print(f"Dataset: {dataset}, N-body: {n_body}, Labels: {labels}")
    print(f"Model: {args.model}, Normalization: Standard MinMax")
    if args.resume:
        print(f"Transfer: {args.resume}")
    print('='*60)
    
    run_staged_training(args, device, config, labels)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Universal Staged Training & Prediction")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=920)
    parser.add_argument('--mode', type=str, default="train", choices=['train', 'predict'],
                       help='Mode: train or predict')
    
    # Data
    parser.add_argument('--dataset', type=str, default="mix")
    parser.add_argument('--n_body', type=str, default="2")
    parser.add_argument('--labels', type=str, default="")
    
    # Model
    parser.add_argument('--model', type=str, default="pamnet", choices=['mxmnet', 'pamnet'])
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=5)
    parser.add_argument('--cutoff_l', type=float, default=5.0)
    parser.add_argument('--cutoff_g', type=float, default=10.0)
    
    # Training
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--patience', type=int, default=20)
    
    # Stage 1
    parser.add_argument('--epochs_s1', type=int, default=300)
    parser.add_argument('--lr_s1', type=float, default=5e-5)
    parser.add_argument('--threshold_high', type=float, default=None)
    
    # Stage 2
    parser.add_argument('--epochs_s2', type=int, default=300)
    parser.add_argument('--lr_s2', type=float, default=3e-5)
    parser.add_argument('--threshold_med', type=float, default=None)
    
    # Stage 3
    parser.add_argument('--epochs_s3', type=int, default=300)
    parser.add_argument('--lr_s3', type=float, default=1e-5)
    
    # Transfer / Resume
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint for transfer learning (train mode)')
    
    # Prediction mode
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint for prediction (predict mode)')
    parser.add_argument('--output_dir', type=str, default='./predictions', help='Output directory for predictions')
    parser.add_argument('--output_prefix', type=str, default='', help='Output file prefix')
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)
    
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(args.gpu)}")
    print(f"Seed: {args.seed}\n")
    
    try:
        if args.mode == 'train':
            train_staged_energy(args, device)
            print("Training completed!\n")
        elif args.mode == 'predict':
            if not args.checkpoint:
                raise ValueError("--checkpoint required for predict mode")
            run_prediction(args, device)
            print("Prediction completed!\n")
    except KeyboardInterrupt:
        print("\n!Interrupted\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n!Error: {type(e).__name__}: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
