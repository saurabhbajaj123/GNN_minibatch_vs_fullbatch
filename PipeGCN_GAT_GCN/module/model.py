from module.layer import *
from torch import nn
from module.sync_bn import SyncBatchNorm
from helper import context as ctx
import dgl



class GNNBase(nn.Module):

    def __init__(self, layer_size, activation, use_pp=False, dropout=0.5, norm='layer', n_linear=0):
        super(GNNBase, self).__init__()
        self.n_layers = len(layer_size) - 1
        self.layers = nn.ModuleList()
        self.activation = activation
        self.use_pp = use_pp
        self.n_linear = n_linear

        if norm is None:
            self.use_norm = False
        else:
            self.use_norm = True
            self.norm = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)


class GraphSAGE(GNNBase):

    def __init__(self, layer_size, activation, use_pp, dropout=0.5, norm='layer', train_size=None, n_linear=0):
        super(GraphSAGE, self).__init__(layer_size, activation, use_pp, dropout, norm, n_linear)
        for i in range(self.n_layers):
            if i < self.n_layers - self.n_linear:
                self.layers.append(GraphSAGELayer(layer_size[i], layer_size[i + 1], use_pp=use_pp))
            else:
                self.layers.append(nn.Linear(layer_size[i], layer_size[i + 1]))
            if i < self.n_layers - 1 and self.use_norm:
                if norm == 'layer':
                    self.norm.append(nn.LayerNorm(layer_size[i + 1], elementwise_affine=True))
                elif norm == 'batch':
                    self.norm.append(SyncBatchNorm(layer_size[i + 1], train_size))
            use_pp = False

    def forward(self, g, feat, in_deg=None):
        h = feat
        for i in range(self.n_layers):
            if i < self.n_layers - self.n_linear:
                if self.training and (i > 0 or not self.use_pp):
                    h = ctx.buffer.update(i, h)
                h = self.dropout(h)
                h = self.layers[i](g, h, in_deg)
            else:
                h = self.dropout(h)
                h = self.layers[i](h)

            if i < self.n_layers - 1:
                if self.use_norm:
                    h = self.norm[i](h)
                h = self.activation(h)

        return h



class GAT(GNNBase):

    def __init__(self, layer_size, activation, use_pp, heads=1, dropout=0.5, norm='layer', train_size=None, n_linear=0):
        super(GAT, self).__init__(layer_size, activation, use_pp, dropout, norm, n_linear)
        self.num_heads = heads
        for i in range(self.n_layers):
            if i < self.n_layers - self.n_linear:
                self.layers.append(dgl.nn.GATConv(layer_size[i], layer_size[i + 1], heads, dropout, dropout))
            else:
                self.layers.append(nn.Linear(layer_size[i], layer_size[i + 1]))
            if i < self.n_layers - 1 and self.use_norm:
                if norm == 'layer':
                    self.norm.append(nn.LayerNorm(layer_size[i + 1], elementwise_affine=True))
                elif norm == 'batch':
                    self.norm.append(SyncBatchNorm(layer_size[i + 1], train_size))

    def forward(self, g, feat, in_deg):
        h = feat
        flops = torch.tensor(0.0)
        for i in range(self.n_layers):
            F_in = h.shape[1]
            num_dst = g.num_nodes('_V')

            if i < self.n_layers - self.n_linear:
                if self.training:
                    if i > 0 or not self.use_pp:
                        h1 = ctx.buffer.update(i, h)
                    else:
                        h1 = h
                        h = h[0:g.num_nodes('_V')]
                    h = self.layers[i](g, (h1, h))
                else:
                    h = self.layers[i](g, h)
                h = h.mean(1)
            else:
                h = self.dropout(h)
                h = self.layers[i](h)
            if i < self.n_layers - 1:
                if self.use_norm:
                    h = self.norm[i](h)
                h = self.activation(h)
            F_out = h.shape[1]
            print(F_in, F_out, in_deg.device, num_dst, self.num_heads)
            flops += gat_flops(F_in, F_out, in_deg.to('cpu'), num_dst, self.num_heads)
            print('flops in the model file', flops)
        return h, flops

    # def forward(self, g, x):
    #     h = x
    #     for i in range(self.n_layers - 1):
    #         h = self.layers[i](g, h)
    #         h = h.flatten(1)
    #         h = self.activation(h)
    #     h = self.layers[-1](g, h)
    #     h = h.mean(1)
    #     return h



# class GCN(GNNBase):

#     def __init__(self, layer_size, activation, use_pp, dropout=0.5, norm='layer', train_size=None, n_linear=0):
#         super(GCN, self).__init__(layer_size, activation, use_pp, dropout, norm, n_linear)
#         for i in range(self.n_layers):
#             if i < self.n_layers - self.n_linear:
#                 self.layers.append(dgl.nn.GraphConv(layer_size[i], layer_size[i + 1], activation=activation))
#             else:
#                 self.layers.append(nn.Linear(layer_size[i], layer_size[i + 1]))
#             if i < self.n_layers - 1 and self.use_norm:
#                 if norm == 'layer':
#                     self.norm.append(nn.LayerNorm(layer_size[i + 1], elementwise_affine=True))
#                 elif norm == 'batch':
#                     self.norm.append(SyncBatchNorm(layer_size[i + 1], train_size))

#     def forward(self, g, features):
#         h = features
#         for i in range(self.n_layers):
#             if i != 0:
#                 h = self.dropout(h)
#             h = self.layers[i](g, h)
#         return h


class GCN(GNNBase):

    def __init__(self, layer_size, activation, use_pp, dropout=0.5, norm='layer', train_size=None, n_linear=0):
        super(GCN, self).__init__(layer_size, activation, use_pp, dropout, norm, n_linear)
        for i in range(self.n_layers):
            if i < self.n_layers - self.n_linear:
                self.layers.append(GCNLayer(layer_size[i], layer_size[i + 1], use_pp=use_pp))
            else:
                self.layers.append(nn.Linear(layer_size[i], layer_size[i + 1]))
            if i < self.n_layers - 1 and self.use_norm:
                if norm == 'layer':
                    self.norm.append(nn.LayerNorm(layer_size[i + 1], elementwise_affine=True))
                elif norm == 'batch':
                    self.norm.append(SyncBatchNorm(layer_size[i + 1], train_size))
            use_pp = False

    def forward(self, g, feat, in_norm=None, out_norm=None):
        h = feat
        for i in range(self.n_layers):
            h = self.dropout(h)
            if i < self.n_layers - self.n_linear:
                if self.training and (i > 0 or not self.use_pp):
                    h = ctx.buffer.update(i, h)
                h = self.layers[i](g, h, in_norm, out_norm)
            else:
                h = self.layers[i](h)

            if i < self.n_layers - 1:
                if self.use_norm:
                    h = self.norm[i](h)
                h = self.activation(h)

        return h


def gat_flops(F_in, F_out, in_deg, num_dst, num_heads):
    num_edges = sum(in_deg)
    return num_heads * (num_edges)*(6*F_in*F_out + 6*F_out + 2) / 1e12
