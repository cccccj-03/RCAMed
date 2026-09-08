import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, GCNConv
from torch_geometric.nn import global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform
import torch.nn as nn
from .GNNConv import GNN_node, GNN_node_Virtualnode

from torch_scatter import scatter_mean


class GNNGraph(torch.nn.Module):

    def __init__(
        self, num_layer=5, emb_dim=300,
        gnn_type='gin', virtual_node=True, residual=False,
        drop_ratio=0.5, JK="last", graph_pooling="mean"
    ):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNNGraph, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        # GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(
                num_layer, emb_dim, JK=JK, drop_ratio=drop_ratio,
                residual=residual, gnn_type=gnn_type
            )
        else:
            self.gnn_node = GNN_node(
                num_layer, emb_dim, JK=JK, drop_ratio=drop_ratio,
                residual=residual, gnn_type=gnn_type
            )

        # Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn=torch.nn.Sequential(
                torch.nn.Linear(emb_dim, 2 * emb_dim),
                torch.nn.BatchNorm1d(2 * emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(2*emb_dim, 1)
            ))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)

        h_graph = self.pool(h_node, batched_data.batch)
        return h_graph
        # return self.graph_pred_linear(h_graph)


class GNN(torch.nn.Module):
    def __init__(
        self, num_tasks, num_layer=5, emb_dim=300,
        gnn_type='gin', virtual_node=True, residual=False,
        drop_ratio=0.5, JK="last", graph_pooling="mean"
    ):
        super(GNN, self).__init__()
        self.model = GNNGraph(
            num_layer, emb_dim, gnn_type,
            virtual_node, residual, drop_ratio, JK, graph_pooling
        )
        self.num_tasks = num_tasks
        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(
                2 * self.model.emb_dim, self.num_tasks
            )
        else:
            self.graph_pred_linear = torch.nn.Linear(
                self.model.emb_dim, self.num_tasks
            )

    def forward(self, batched_data):
        h_graph = self.model(batched_data)
        return self.graph_pred_linear(h_graph)


class GCN_SW(GCNConv):
    # pycharm让我加的，我没看懂
    def edge_update(self):
        pass

    def __init__(self, in_channels, out_channels, edge_types, normalize=False, bias=True):
        super(GCN_SW, self).__init__(in_channels, out_channels, normalize, bias)
        self.edge_types = edge_types

        self.edge_embedding = nn.Embedding(edge_types, 1)
        self.ddi_weight = nn.Parameter(torch.tensor(0.1))
        self.init_weight()

    def forward(self, x, edge_index, edge_weight = None,
                ddi_weight = None):
        x = self.lin(x)

        edge_weight = self.edge_embedding(edge_weight)
#         ddi_weight = ddi_weight.unsqueeze(1) * self.ddi_weight
       #  edge_weight = edge_weight - ddi_weight

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)

        if self.bias is not None:
            out = out + self.bias

        # print(self.edge_embedding.weight.data)
        return out#

    def init_weight(self):
        weight = 1 / self.edge_types
        for i in range(self.edge_types):
            self.edge_embedding.weight.data[i].copy_(weight * i)


class GCN_MF(GCNConv):
    # pycharm让我加的，我没看懂
    def edge_update(self):
        pass

    def __init__(self, in_channels, out_channels, edge_types, normalize=False, bias=True):
        super(GCN_MF, self).__init__(in_channels, out_channels, normalize, bias)
        self.edge_types = edge_types
        self.edge_embedding = nn.Embedding(edge_types, in_channels)
        self.init_weight()

    def forward(self, x, edge_index, edge_weight = None) :
        x = self.lin(x)

        edge_weight = self.edge_embedding(edge_weight)

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)


        if self.bias is not None:
            out = out + self.bias

        # print(self.edge_embedding.weight.data)
        return out# , dim=0, keepdim=True).unsqueeze(0)

    def message(self, x_j, edge_weight):
        result = edge_weight * x_j
        return result

    def init_weight(self):
        weight = 0.8 / self.edge_types
        for i in range(self.edge_types):
            self.edge_embedding.weight.data[i].copy_(weight * i)

if __name__ == '__main__':
    GNN(num_tasks=10)
