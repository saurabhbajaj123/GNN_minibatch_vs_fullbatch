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
import wandb
wandb.login()

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

os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["DGLDEFAULTDIR"] = "/work/sbajaj_umass_edu/.dgl"
os.environ["DGL_DOWNLOAD_DIR"] = "/work/sbajaj_umass_edu/.dgl"


root = "/work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/dataset"

args = create_parser()

def sample(graph, nids, num_layers):
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(num_layers)

    dataloader = dgl.dataloading.DataLoader(
        graph, nids, sampler,
        batch_size=2048, shuffle=False, drop_last=False, num_workers=1)

    graph_array = []
    for input_nodes, output_nodes, blocks in dataloader:
        for block in blocks:
            graph_array.append(dgl.block_to_graph(block))
        break

    merged_sub = dgl.merge(graph_array)

    a = merged_sub.ndata['_ID']['_N_dst']
    b = merged_sub.ndata['_ID']['_N_src']

    ids = torch.cat((a, b)).unique()

    sg = graph.subgraph(ids, relabel_nodes=True)

    return sg
def node_ids_to_keep(graph, nids, num_layers):
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(num_layers)

    dataloader = dgl.dataloading.DataLoader(
        graph, nids, sampler,
        batch_size=2048, shuffle=False, drop_last=False, num_workers=1)

    graph_array = []
    id_array = []
    for input_nodes, output_nodes, blocks in dataloader:
        print(f"input_nodes = {input_nodes}")
        id_array.append(input_nodes)
        # dgl.block_to_graph(blocks[0])
        # for block in blocks:
        #     graph_array.append(dgl.block_to_graph(block))
        #     print(graph_array[-1].ndata)
        #     break
        # break

    # merged_sub = dgl.merge(graph_array)

    # a = merged_sub.ndata['_ID']['_N_dst']
    # b = merged_sub.ndata['_ID']['_N_src']

    ids = torch.cat(id_array).unique()
    return ids
#############################################################################################

dataset = AsNodePredDataset(DglNodePropPredDataset(args.dataset, root=root))
# print(dataset.train_idx)
# print(dataset.val_idx)
# print(dataset.test_idx)

# print(type(dataset.train_idx))
# print(random.sample((dataset.train_idx).tolist(), 3)) 

frac = 0.01
sampled_train_ids = random.sample((dataset.train_idx).tolist(), int(frac*len(dataset.train_idx)))
print(len(sampled_train_ids))
sampled_val_ids = random.sample((dataset.val_idx).tolist(), int(frac*len(dataset.val_idx)))
print(len(sampled_val_ids))
sampled_test_ids = random.sample((dataset.test_idx).tolist(), int(frac*len(dataset.test_idx)))
print(len(sampled_test_ids))

sample_nids = sampled_train_ids + sampled_val_ids + sampled_test_ids
print(f"total sampled nids = {len(sample_nids)}")
# # # device = "cuda" if torch.cuda.is_available() else "cpu"
# # # device = "cpu"
graph = dataset[0]

print(f"len(graph.ndata['train_mask']) = {len(graph.ndata['train_mask'])}")

print(f"len(graph.ndata['train_mask'][sample_nids]) {len(graph.ndata['train_mask'][sample_nids])}")

subgraph_ids = node_ids_to_keep(graph, sample_nids, 2)

print(f"len subgraph_ids = {len(subgraph_ids)}")
# train_tiny_ids = node_ids_to_keep(graph, sampled_train_ids, 3)
# valid_tiny_ids = node_ids_to_keep(graph, sampled_val_ids, 3)
# test_tiny_ids = node_ids_to_keep(graph, sampled_test_ids, 3)

# ids = torch.cat([train_tiny_ids, valid_tiny_ids, test_tiny_ids]).unique()

# print(ids)

# orgi_train_idx = orgi_train_idx[ids]
# orgi_valid_idx = orgi_valid_idx[ids]
# orgi_test_idx = orgi_test_idx[ids]


sg = graph.subgraph(subgraph_ids, relabel_nodes=True)


print(f"len of sg train mask = {len(sg.ndata['train_mask'])}")

dgl.save_graphs("arxiv_subgraph.bin", sg)

# sg.ndata['train_mask'] = graph.ndata['train_mask'][sample_nids]
# sg.ndata['valid_mask'] = graph.ndata['valid_mask'][sample_nids]
# sg.ndata['test_mask'] = graph.ndata['test_mask'][sample_nids]




# num_layers = args.n_layers
# num_hidden = args.n_hidden



# train_sg = sample(graph, train_nids, args.num_layers)
# valid_sg = sample(graph, valid_nids, args.num_layers)
# test_sg = sample(graph, test_nids, args.num_layers)

# print(f"train_sg = {train_sg}")
# print(f"train_sg.ndata = {train_sg.ndata}")

# # print(merged_sub)
# dgl.save_graphs("arxiv_subgraph_train.bin", train_sg)
# dgl.save_graphs("arxiv_subgraph_valid.bin", valid_sg)
# dgl.save_graphs("arxiv_subgraph_test.bin", test_sg)

# loaded_graph_train = dgl.load_graphs("arxiv_subgraph_train.bin")
# loaded_graph_valid = dgl.load_graphs("arxiv_subgraph_valid.bin")
# loaded_graph_test = dgl.load_graphs("arxiv_subgraph_test.bin")
# print(loaded_graph_train[0][0].ndata)
# print(loaded_graph_valid[0][0].ndata)
# print(loaded_graph_test[0][0].ndata)


#############################################################################################

# tiny_graph = dgl.graph(([0, 1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 7, 7, 8, 8, 0]))
# print(f"tiny_graph = {tiny_graph}")
# print(f"tiny_graph.nodes() = {tiny_graph.nodes()}")


# train_tiny_sg = sample(tiny_graph, [0], 2)
# valid_tiny_sg = sample(tiny_graph, [2], 2)
# test_tiny_sg = sample(tiny_graph, [1, 3, 4], 2)

# print(train_tiny_sg, train_tiny_sg.ndata, train_tiny_sg.nodes())
# print(valid_tiny_sg, valid_tiny_sg.ndata, valid_tiny_sg.nodes())
# print(test_tiny_sg, test_tiny_sg.ndata, test_tiny_sg.nodes())

# train_ids = train_tiny_sg.ndata[dgl.NID]
# valid_ids = valid_tiny_sg.ndata[dgl.NID]
# test_ids = test_tiny_sg.ndata[dgl.NID]

# final_graph = dgl.merge([train_tiny_sg, valid_tiny_sg, test_tiny_sg])

# print(f"final_graph.ndata = {final_graph.ndata}")
# print(train_tiny_sg, train_tiny_sg.ndata)
# print(valid_tiny_sg, valid_tiny_sg.ndata)
# print(test_tiny_sg, test_tiny_sg.ndata)

# # print(final_graph.ndata['train_nids'])

# final_graph.ndata['train_nids'] = train_ids
# final_graph.ndata['valid_nids'] = valid_ids
# final_graph.ndata['test_nids'] = test_ids

# print(final_graph.ndata)

#############################################################################################



# train_tiny_ids = node_ids_to_keep(tiny_graph, [0], 3)
# valid_tiny_ids = node_ids_to_keep(tiny_graph, [2], 2)
# test_tiny_ids = node_ids_to_keep(tiny_graph, [1, 3, 4], 2)

# ids = torch.cat([train_tiny_ids, valid_tiny_ids, test_tiny_ids]).unique()


# train_mask = numpy.zeros(len(ids))
# valid_mask = numpy.zeros(len(ids))
# test_mask = numpy.zeros(len(ids))

# train_mask[train_tiny_ids.numpy()] = 1
# valid_mask[valid_tiny_ids.numpy()] = 1
# test_mask[test_tiny_ids.numpy()] = 1


# sg = tiny_graph.subgraph(ids, relabel_nodes=True)

# sg.ndata['train_mask'] = train_mask
# sg.ndata['valid_mask'] = valid_mask
# sg.ndata['test_mask'] = test_mask

# print(sg.ndata)