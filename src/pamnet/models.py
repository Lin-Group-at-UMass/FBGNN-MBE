import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.nn import global_mean_pool, global_add_pool, radius
from torch_geometric.utils import remove_self_loops
from torch_scatter import scatter
from .layers import Global_MessagePassing, Local_MessagePassing, Local_MessagePassing_s,  BesselBasisLayer, SphericalBasisLayer, MLP

class Config(object):
    def __init__(self, dim, n_layer, cutoff_l, cutoff_g, dataset=None, mode='energy', use_charge=False):
        self.dim = dim
        self.n_layer = n_layer
        self.cutoff_l = cutoff_l
        self.cutoff_g = cutoff_g
        self.cutoff = cutoff_g
        self.dataset = dataset
        self.mode = mode
        self.use_charge = use_charge

class PAMNet(nn.Module):
    def __init__(self, config: Config, mode: str = 'energy', use_charge: bool = False, num_spherical=7, num_radial=6, envelope_exponent=5):
        super(PAMNet, self).__init__()

        self.mode = mode
        self.use_charge = use_charge
        self.dim = config.dim
        self.n_layer = config.n_layer
        self.cutoff_l = config.cutoff_l
        self.cutoff_g = config.cutoff_g
        self.cutoff = config.cutoff

        self.embeddings = nn.Parameter(torch.ones((5, self.dim)))

        self.rbf_g = BesselBasisLayer(16, self.cutoff_g, envelope_exponent)
        self.rbf_l = BesselBasisLayer(16, self.cutoff_l, envelope_exponent)
        self.sbf = SphericalBasisLayer(num_spherical, num_radial, self.cutoff_l, envelope_exponent)

        self.mlp_rbf_g = MLP([16, self.dim])
        self.mlp_rbf_l = MLP([16, self.dim])
        self.mlp_sbf1 = MLP([num_spherical * num_radial, self.dim])
        self.mlp_sbf2 = MLP([num_spherical * num_radial, self.dim])

        input_feature_dim = self.dim
        if self.use_charge:
            input_feature_dim += 1
        self.feature_projection = MLP([input_feature_dim, self.dim])

        if self.mode == 'eda':
            self.output_head = MLP([self.dim, 3])
        else:
            # self.output_head = MLP([self.dim, 1])
            self.output_head = nn.Linear(self.dim, 1)

        self.global_layer = torch.nn.ModuleList()
        for _ in range(config.n_layer):
            self.global_layer.append(Global_MessagePassing(config))

        self.local_layer = torch.nn.ModuleList()
        for _ in range(config.n_layer):
            self.local_layer.append(Local_MessagePassing(config))

        self.init()
        

    def init(self):
        stdv = math.sqrt(3)
        self.embeddings.data.uniform_(-stdv, stdv)

    def indices(self, edge_index, num_nodes):
        row, col = edge_index
        value = torch.arange(row.size(0), device=row.device)
        adj_t = SparseTensor(row=col, col=row, value=value, sparse_sizes=(num_nodes, num_nodes))
        adj_t_row = adj_t[row]
        num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)
        idx_i = col.repeat_interleave(num_triplets)
        idx_j = row.repeat_interleave(num_triplets)
        idx_k = adj_t_row.storage.col()
        mask = idx_i != idx_k
        idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]
        idx_kj = adj_t_row.storage.value()[mask]
        idx_ji = adj_t_row.storage.row()[mask]
        adj_t_col = adj_t[col]
        num_pairs = adj_t_col.set_value(None).sum(dim=1).to(torch.long)
        idx_i_pair = row.repeat_interleave(num_pairs)
        idx_j1_pair = col.repeat_interleave(num_pairs)
        idx_j2_pair = adj_t_col.storage.col()
        mask_j = idx_j1_pair != idx_j2_pair
        idx_i_pair, idx_j1_pair, idx_j2_pair = idx_i_pair[mask_j], idx_j1_pair[mask_j], idx_j2_pair[mask_j]
        idx_ji_pair = adj_t_col.storage.row()[mask_j]
        idx_jj_pair = adj_t_col.storage.value()[mask_j]
        return idx_i, idx_j, idx_k, idx_kj, idx_ji, idx_i_pair, idx_j1_pair, idx_j2_pair, idx_jj_pair, idx_ji_pair

    def forward(self, data):
        x_raw = data.x
        edge_index_l = data.edge_index
        batch = data.batch
        pos = data.pos

        h_embedding = torch.index_select(self.embeddings, 0, x_raw.long())

        if self.use_charge:
            if hasattr(data, 'q'):
                h_initial = torch.cat([h_embedding, data.q.view(-1, 1)], dim=1)
            elif hasattr(data, 'charges'):
                h_initial = torch.cat([h_embedding, data.charges.view(-1, 1)], dim=1)
            else:
                h_initial = h_embedding
        else:
            h_initial = h_embedding

        h = self.feature_projection(h_initial)

        # Global graph
        row_g, col_g = radius(pos, pos, self.cutoff_g, batch, batch, max_num_neighbors=1000)
        edge_index_g = torch.stack([row_g, col_g], dim=0)
        edge_index_g, _ = remove_self_loops(edge_index_g)
        
        # Check global edges
        if edge_index_g.size(1) == 0:
            print("!! WARNING: No global edges found (radius graph is empty). Check cutoff_g or pos scaling.")

        j_g, i_g = edge_index_g
        dist_g = (pos[i_g] - pos[j_g]).pow(2).sum(dim=-1).sqrt()

        # Local graph
        if edge_index_l is None:
            raise RuntimeError("edge_index_l is None! Model requires local edges.")
            
        edge_index_l, _ = remove_self_loops(edge_index_l)
        j_l, i_l = edge_index_l
        dist_l = (pos[i_l] - pos[j_l]).pow(2).sum(dim=-1).sqrt()

        # Triplet/Angle indices
        idx_i, idx_j, idx_k, idx_kj, idx_ji, idx_i_pair, idx_j1_pair, idx_j2_pair, idx_jj_pair, idx_ji_pair = self.indices(edge_index_l, num_nodes=h.size(0))

        # Check triplets
        if idx_kj.size(0) == 0:
            print("!! WARNING: No triplets found. Message passing might be failing.")

        pos_ji, pos_kj = pos[idx_j] - pos[idx_i], pos[idx_k] - pos[idx_j]
        a = (pos_ji * pos_kj).sum(dim=-1)
        b = torch.cross(pos_ji, pos_kj, dim=-1).norm(dim=-1)
        angle2 = torch.atan2(b, a)

        pos_ji_pair, pos_jj_pair = pos[idx_j1_pair] - pos[idx_i_pair], pos[idx_j2_pair] - pos[idx_j1_pair]
        a = (pos_ji_pair * pos_jj_pair).sum(dim=-1)
        b = torch.cross(pos_ji_pair, pos_jj_pair, dim=-1).norm(dim=-1)
        angle1 = torch.atan2(b, a)

        rbf_l = self.mlp_rbf_l(self.rbf_l(dist_l))
        rbf_g = self.mlp_rbf_g(self.rbf_g(dist_g))
        sbf1 = self.mlp_sbf1(self.sbf(dist_l, angle1, idx_jj_pair))
        sbf2 = self.mlp_sbf2(self.sbf(dist_l, angle2, idx_kj))

        for layer in range(self.n_layer):
            h, out_g, _ = self.global_layer[layer](h, rbf_g, edge_index_g)
            h, out_l, _ = self.local_layer[layer](h, rbf_l, sbf2, sbf1,
                                                  idx_kj, idx_ji, idx_jj_pair, idx_ji_pair, edge_index_l)

        if self.mode == 'eda':
            graph_representation = global_add_pool(h, batch)
        else:
            graph_representation = global_add_pool(h, batch)

        output = self.output_head(graph_representation)

        if self.mode == 'gradient':
            predicted_energy = output.view(-1)
            
            if not pos.requires_grad:
                pos.requires_grad_(True)
            
            try:
                grad_outputs = torch.autograd.grad(
                    outputs=predicted_energy.sum(),
                    inputs=pos,
                    create_graph=True,    
                    retain_graph=True, 
                    allow_unused=False 
                )
                forces = -grad_outputs[0]
            except RuntimeError as e:
                print("\n!!! GRADIENT COMPUTATION FAILED !!!")
                print(f"Error: {e}")
                print("Diagnostic info:")
                print(f"  Pos requires grad: {pos.requires_grad}")
                print(f"  Edge Index Global size: {edge_index_g.size()}")
                print(f"  Edge Index Local size: {edge_index_l.size()}")
                print(f"  Triplets count: {idx_kj.size(0)}")
                raise e
            
            return predicted_energy, forces
            
        elif self.mode == 'energy':
            return output.view(-1)
        elif self.mode == 'eda':
            return output
            
