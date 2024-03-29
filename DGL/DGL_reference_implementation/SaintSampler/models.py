import argparse
import json
import logging
import os
import sys
import pickle

import dgl
from dgl.nn import GATConv, SAGEConv
import dgl.nn as dglnn
import torch
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset
import time 
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


import random
import wandb
wandb.login()

import torch.nn as nn
import torch.nn.functional as F
# from dgl.nn import SAGEConv
import tqdm
import sklearn.metrics


import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



class SAGE(nn.Module):
    def __init__(
        self, in_feats, n_hidden, n_classes, n_layers, dropout, activation, aggregator_type='mean'
    ):
        super(SAGE, self).__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type=aggregator_type))
        for _ in range(n_layers - 2):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type=aggregator_type))
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type=aggregator_type))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, g, x):
        h = x
        for l, conv in enumerate(self.layers):
            h = conv(g, h)
            # print("self.activation = {}".format(type(self.activation)))
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
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
