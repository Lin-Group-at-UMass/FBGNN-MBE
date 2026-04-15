import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool, radius_graph
from torch_geometric.nn.models.schnet import GaussianSmearing, ShiftedSoftplus, InteractionBlock
from torch_geometric.nn import DimeNet as PyGDimeNet
from torch_geometric.nn import DimeNetPlusPlus as PyGDimeNetPP
from torch_geometric.data import Data, Batch

# SchNet
class SchNetStudent(nn.Module):
    def __init__(self, in_dim=100, hidden_dim=128, out_dim=1, num_layers=3,
                 num_filters=128, cutoff=10.0, **kwargs):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.cutoff = cutoff
        self.num_layers = num_layers
        
        # Atomic embedding
        self.embedding = nn.Embedding(in_dim, hidden_dim)
        
        # Distance expansion
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_filters)
        
        # Interaction blocks
        self.interactions = nn.ModuleList([
            InteractionBlock(hidden_dim, num_filters, num_filters, cutoff)
            for _ in range(num_layers)
        ])
        
        # Output layers
        self.lin1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.act = ShiftedSoftplus()
        self.lin2 = nn.Linear(hidden_dim // 2, out_dim)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.embedding.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()
        nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, data):
        return self.forward_with_features(data)[0]
    
    def forward_with_features(self, data, layer_idx=-1):
        z = data.x.long().squeeze(-1) if data.x.dim() > 1 else data.x.long()
        pos = data.pos
        batch = data.batch
        
        # Build graph
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch, max_num_neighbors=32)
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        edge_attr = self.distance_expansion(edge_weight)
        
        # Initial embedding
        h = self.embedding(z)
        
        # Store features per layer
        layer_features = [h.clone()]
        
        # Interaction blocks
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)
            layer_features.append(h.clone())
        
        # Get node features from specified layer
        if layer_idx == -1:
            node_feat = layer_features[-1]
        else:
            idx = min(layer_idx, len(layer_features) - 1)
            node_feat = layer_features[idx]
        
        # Output
        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)
        out = global_add_pool(h, batch).squeeze(-1)
        
        # Graph features
        graph_feat = global_mean_pool(node_feat, batch)
        
        return out, node_feat, graph_feat


# DimeNet
class DimeNetStudent(nn.Module):
    def __init__(self, in_dim=100, hidden_dim=128, out_dim=1, num_layers=3,
                 num_bilinear=8, num_spherical=7, num_radial=6, cutoff=5.0, **kwargs):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.cutoff = cutoff
        self.in_dim = in_dim
        
        # Use PyG's DimeNet for predictions
        self._dimenet = PyGDimeNet(
            hidden_channels=hidden_dim,
            out_channels=out_dim,
            num_blocks=num_layers,
            num_bilinear=num_bilinear,
            num_spherical=num_spherical,
            num_radial=num_radial,
            cutoff=cutoff,
        )
        
        # Separate node encoder for feature distillation, [num_nodes, hidden_dim] features
        self.node_encoder = nn.Sequential(
            nn.Embedding(in_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for module in self.node_encoder:
            if isinstance(module, (nn.Linear, nn.Embedding)):
                if hasattr(module, 'weight'):
                    nn.init.xavier_uniform_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data.fill_(0)
    
    def forward(self, data):
        return self.forward_with_features(data)[0]
    
    def forward_with_features(self, data, layer_idx=-1):
        z = data.x.long().squeeze(-1) if data.x.dim() > 1 else data.x.long()
        pos = data.pos
        batch = data.batch
        num_nodes = z.size(0)
        
        # Get predictions
        out = self._dimenet(z, pos, batch).squeeze(-1)
        
        # Create node-level features for distillation
        # Use the node encoder on atomic numbers
        node_feat = self.node_encoder[0](z)  # Embedding
        for layer in self.node_encoder[1:]:
            node_feat = layer(node_feat)
        
        # Ensure correct shape [num_nodes, hidden_dim]
        assert node_feat.shape == (num_nodes, self.hidden_dim), \
            f"Node feat shape {node_feat.shape} != expected ({num_nodes}, {self.hidden_dim})"
        
        # Graph-level features
        graph_feat = global_mean_pool(node_feat, batch)
        
        return out, node_feat, graph_feat


# DimeNet++
class DimeNetPlusPlusStudent(nn.Module):
    def __init__(self, in_dim=100, hidden_dim=128, out_dim=1, num_layers=3,
                 int_emb_size=64, basis_emb_size=8, out_emb_size=256,
                 num_spherical=7, num_radial=6, cutoff=5.0, **kwargs):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.cutoff = cutoff
        self.in_dim = in_dim
        
        # Use PyG's DimeNet++ for predictions
        self._dimenet_pp = PyGDimeNetPP(
            hidden_channels=hidden_dim,
            out_channels=out_dim,
            num_blocks=num_layers,
            int_emb_size=int_emb_size,
            basis_emb_size=basis_emb_size,
            out_emb_channels=out_emb_size,
            num_spherical=num_spherical,
            num_radial=num_radial,
            cutoff=cutoff,
        )
        
        # Separate node encoder for feature distillation, [num_nodes, hidden_dim] features
        self.node_encoder = nn.Sequential(
            nn.Embedding(in_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for module in self.node_encoder:
            if isinstance(module, (nn.Linear, nn.Embedding)):
                if hasattr(module, 'weight'):
                    nn.init.xavier_uniform_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data.fill_(0)
    
    def forward(self, data):
        return self.forward_with_features(data)[0]
    
    def forward_with_features(self, data, layer_idx=-1):
        z = data.x.long().squeeze(-1) if data.x.dim() > 1 else data.x.long()
        pos = data.pos
        batch = data.batch
        num_nodes = z.size(0)
        
        # Get predictions from DimeNet++
        out = self._dimenet_pp(z, pos, batch).squeeze(-1)
        
        # Create node-level features for distillation
        # Use the node encoder on atomic numbers
        node_feat = self.node_encoder[0](z)  # Embedding
        for layer in self.node_encoder[1:]:
            node_feat = layer(node_feat)
        
        # Ensure correct shape [num_nodes, hidden_dim]
        assert node_feat.shape == (num_nodes, self.hidden_dim), \
            f"Node feat shape {node_feat.shape} != expected ({num_nodes}, {self.hidden_dim})"
        
        # Graph-level features
        graph_feat = global_mean_pool(node_feat, batch)
        
        return out, node_feat, graph_feat



# ViSNet
class ViSNetStudent(nn.Module):
    def __init__(self, in_dim=100, hidden_dim=128, out_dim=1, num_layers=3,
                 lmax=1, vecnorm_type='none', trainable_vecnorm=False,
                 num_heads=8, num_rbf=32, cutoff=5.0, **kwargs):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.cutoff = cutoff
        
        from torch_geometric.nn import ViSNet as _ViSNet
        
        self._model = _ViSNet(
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            num_rbf=num_rbf,
            cutoff=cutoff,
            lmax=lmax,
            vecnorm_type=vecnorm_type,
            trainable_vecnorm=trainable_vecnorm,
            num_heads=num_heads,
        )
        
        # HCapture intermediate features
        self._node_features = None
        self._register_hooks()
    
    def _register_hooks(self):
        def hook_fn(module, input, output):
            # ViSNet returns tuple (scalar, vector)
            if isinstance(output, tuple):
                self._node_features = output[0].clone()
            elif isinstance(output, torch.Tensor):
                self._node_features = output.clone()
        
        # Hook the representation or final vis_mp layer.
        if hasattr(self._model, 'representation'):
            if hasattr(self._model.representation, 'vis_mp_layers'):
                layers = self._model.representation.vis_mp_layers
                if len(layers) > 0:
                    layers[-1].register_forward_hook(hook_fn)
    
    def forward(self, data):
        return self.forward_with_features(data)[0]
    
    def forward_with_features(self, data, layer_idx=-1):
        z = data.x.long().squeeze(-1) if data.x.dim() > 1 else data.x.long()
        pos = data.pos
        batch = data.batch
        
        self._node_features = None
        
        visnet_out = self._model(z, pos, batch)
        out = visnet_out[0].squeeze(-1) if isinstance(visnet_out, tuple) else visnet_out.squeeze(-1)
        
        # Get node features from hook
        if self._node_features is not None:
            node_feat = self._node_features
            if node_feat.dim() == 1:
                node_feat = node_feat.unsqueeze(-1)
            # Ensure feature dim matches hidden_dim
            if node_feat.shape[-1] != self.hidden_dim:
                # Project to hidden_dim
                if not hasattr(self, '_feat_proj'):
                    self._feat_proj = nn.Linear(node_feat.shape[-1], self.hidden_dim).to(node_feat.device)
                node_feat = self._feat_proj(node_feat)
            graph_feat = global_mean_pool(node_feat, batch)
        else:
            # Fallback
            node_feat = torch.zeros(z.size(0), self.hidden_dim, device=z.device)
            graph_feat = global_mean_pool(node_feat, batch)
        
        return out, node_feat, graph_feat


# Function to create student model
def get_student_model(name, in_dim=100, hidden_dim=128, num_layers=3, edge_dim=None, **kwargs):
    name = name.lower().replace(' ', '').replace('_', '')
    
    if name == 'schnet':
        return SchNetStudent(in_dim=in_dim, hidden_dim=hidden_dim, num_layers=num_layers, **kwargs)
    elif name == 'dimenet':
        return DimeNetStudent(in_dim=in_dim, hidden_dim=hidden_dim, num_layers=num_layers, **kwargs)
    elif name in ['dimenet++', 'dimenetplusplus', 'dimenetpp']:
        return DimeNetPlusPlusStudent(in_dim=in_dim, hidden_dim=hidden_dim, num_layers=num_layers, **kwargs)
    elif name == 'visnet':
        return ViSNetStudent(in_dim=in_dim, hidden_dim=hidden_dim, num_layers=num_layers, **kwargs)
    else:
        raise ValueError(f"Unknown model: {name}. Available: schnet, dimenet, dimenet++, visnet")
