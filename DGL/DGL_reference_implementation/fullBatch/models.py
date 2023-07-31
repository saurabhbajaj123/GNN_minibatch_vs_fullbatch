import os
os.environ['DGLBACKEND'] = 'pytorch'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import random
import wandb
wandb.login()
import dgl
import dgl.data
import dgl.nn.pytorch as dglnn
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.nn import GATConv


class SAGE(nn.Module):
    def __init__(
        self, in_feats, n_classes, n_hidden, n_layers, activation, dropout, agg
    ):
        print(activation)
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, agg))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, agg))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, agg))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        # print(self.activation)

    def forward(self, g, x):
        h = x
        for l, conv in enumerate(self.layers):
            h = conv(g, h)
            # print("self.activation = {}".format(type(self.activation)))
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, x, batch_size, device):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.
        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        h = x
        for l, conv in enumerate(self.layers):
            h = conv(g, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)

        return h

class GAT(nn.Module):
    def __init__(
        self, in_feats, n_hidden, n_classes, n_layers, num_heads
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.num_heads = num_heads

        self.layers = nn.ModuleList()
        self.layers.append(GATConv(in_feats, n_hidden, num_heads=num_heads))
        for _ in range(n_layers - 2):
            self.layers.append(GATConv(n_hidden*num_heads, n_hidden, num_heads=num_heads))
        self.layers.append(GATConv(n_hidden*num_heads, n_classes, num_heads=1))


    def forward(self, g, x):
        h = x
        for i in range(self.n_layers - 1):
            h = self.layers[i](g, h)
            h = h.flatten(1)
        h = self.layers[-1](g, h)
        h = h.mean(1)
        return h




class GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GCN
        self.layers.append(dglnn.GraphConv(in_feats, n_hidden, activation=F.relu))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.GraphConv(n_hidden, n_hidden, activation=F.relu))

        self.layers.append(dglnn.GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h
