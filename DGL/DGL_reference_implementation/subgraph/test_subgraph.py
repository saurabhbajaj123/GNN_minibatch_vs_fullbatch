import argparse
import json
import logging
import os
import sys
import pickle

import dgl
import dgl.data
import torch
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset
import time 
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from dgl.data import AsNodePredDataset

import random
# import wandb
# wandb.login()

import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv, GraphConv, GATConv
import tqdm
import sklearn.metrics
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
from parser import create_parser
import warnings
warnings.filterwarnings("ignore")

import dgl
import torch
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset
import time 

# os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["DGLDEFAULTDIR"] = "/home/ubuntu/gnn_mini_vs_full/.dgl"
os.environ["DGL_DOWNLOAD_DIR"] = "/home/ubuntu/gnn_mini_vs_full/.dgl"


# root = "/home/ubuntu/gnn_mini_vs_full/GNN_minibatch_vs_fullbatch/dataset"
# root = "/home/ubuntu/gnn_mini_vs_full/GNN_minibatch_vs_fullbatch/DGL/DGL_reference_implementation/dataset"
root = "/home/ubuntu/gnn_mini_vs_full/GNN_minibatch_vs_fullbatch/DGL/DGL_reference_implementation/dataset/sub_dataset_folder"


def load_subgraph(dataset_path):
    g, _ = dgl.load_graphs(dataset_path)
    g = g[0]
    g.ndata['label'] = g.ndata['label'].to(torch.int64)
    n_feat = g.ndata['feat'].shape[1]
    print("train_mask shape = {}".format(g.ndata['train_mask'].shape))
    print("label shape = {}".format(g.ndata['label'].shape))
    if g.ndata['label'].dim() == 1:
        # n_class = g.ndata['label'].max().item() + 1
        n_class = int(torch.max(torch.unique(g.ndata['label'][torch.logical_not(torch.isnan(g.ndata['label']))])).item()) + 1
    else:
        n_class = g.ndata['label'].shape[1]
    return g, n_feat, n_class

# path = '/home/ubuntu/gnn_mini_vs_full/GNN_minibatch_vs_fullbatch/DGL/DGL_reference_implementation/subgraph/ogbn-papers100M_frac_100.0_hops_3_subgraph.bin'
path = '/home/ubuntu/gnn_mini_vs_full/MGG-OSDI23-AE/dgl_pydirect_internal/graphdata/com-orkut.mtx_dgl_graph.bin'
g, n_feat, n_class = load_subgraph(path)


print(g)
print(g.ndata['label'])
print(torch.unique(g.ndata['label']))
tot = g.num_nodes()
u, v = g.edges()
print(u, v)


# isolated_nodes = ((g.in_degrees() == 0) & (g.out_degrees() == 0)).nonzero().squeeze(1)
# g.remove_nodes(isolated_nodes)
