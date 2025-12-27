import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (GATv2Conv, GINConv, TransformerConv, global_mean_pool, SchNet, DimeNet, DimeNetPlusPlus, ViSNet)
from torch_geometric.utils import scatter


# SchNet
class SchNetStudent(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=1, num_layers=2, **kwargs):
        super().__init__()
        self.schnet = SchNet(
            hidden_channels=hidden_dim,
            num_filters=128, 
            num_interactions=num_layers,
            readout='add'
        )

    def forward(self, data):
        return self.forward_with_features(data)[0]

    def forward_with_features(self, data):
        z, pos, batch = data.x.long(), data.pos, data.batch
        schnet_output = self.schnet(z, pos, batch)
        out = schnet_output.view(-1)
        feat_student = schnet_output
        
        return out, feat_student, None


# DimeNet and DimeNet++ 
class DimeNetStudent(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=1, num_layers=2, **kwargs):
        super().__init__()
        self.dimenet = DimeNet(
            hidden_channels=hidden_dim,
            out_channels=out_dim, # DimeNet takes an explicit out_channels argument
            num_blocks=num_layers,
            num_bilinear=8,
            num_spherical=7,
            num_radial=6
        )

    def forward(self, data):
        return self.forward_with_features(data)[0]

    def forward_with_features(self, data):
        z = data.x.long()
        pos = data.pos
        batch = data.batch
        out = self.dimenet(z, pos, batch).view(-1)
        return out, None, None

class DimeNetPlusPlusStudent(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=1, num_layers=2, **kwargs):
        super().__init__()
        self.dimenet_pp = DimeNetPlusPlus(
            hidden_channels=hidden_dim,
            out_channels=out_dim, # DimeNet++ also takes out_channels
            num_blocks=num_layers,
            int_emb_size=64,
            basis_emb_size=8,
            out_emb_channels=256,
            num_spherical=7,
            num_radial=6
        )

    def forward(self, data):
        return self.forward_with_features(data)[0]

    def forward_with_features(self, data):
        z = data.x.long()
        pos = data.pos
        batch = data.batch
        out = self.dimenet_pp(z, pos, batch).view(-1)
        return out, None, None

# ViSNetStudent
class ViSNetStudent(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=1, num_layers=2, **kwargs):
        super().__init__()
        self.visnet = ViSNet(
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            # ViSNet has its own internal output layer
        )

    def forward(self, data):
        return self.forward_with_features(data)[0]

    def forward_with_features(self, data):
        z = data.x.long()
        pos = data.pos
        batch = data.batch
        
        visnet_output = self.visnet(z, pos, batch)[0]
        out = visnet_output.view(-1)
        feat_student = visnet_output
        
        return out, feat_student, None


def get_student_model(name, in_dim, hidden_dim=64, num_layers=2, edge_dim=None):
    name = name.lower()
    if name == "schnet":
        return SchNetStudent(in_dim=in_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    elif name == "dimenet":
        return DimeNetStudent(in_dim=in_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    elif name == "dimenet++":
        return DimeNetPlusPlusStudent(in_dim=in_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    elif name == "visnet":
        return ViSNetStudent(in_dim=in_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    else:
        raise ValueError(f"Unknown student model: {name}")

