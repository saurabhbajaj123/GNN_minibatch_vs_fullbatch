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

args = create_parser()

print(f"args = {args}")
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
    dataloader = dgl.dataloading.DataLoader(graph, nids, sampler,batch_size=2048, shuffle=False, drop_last=False, num_workers=1)
    graph_array = []
    id_array = []
    for input_nodes, output_nodes, blocks in dataloader:
        id_array.append(input_nodes)
    ids = torch.cat(id_array).unique()
    return ids

def func(dataset, frac, n_layers):
    print("frac, n_layers = {} {}".format(frac, n_layers))
    graph = dataset[0]
    sampled_train_ids = random.sample((dataset.train_idx).tolist(), int(frac*len(dataset.train_idx)))
    sampled_val_ids = random.sample((dataset.val_idx).tolist(), int(frac*len(dataset.val_idx)))
    sampled_test_ids = random.sample((dataset.test_idx).tolist(), int(frac*len(dataset.test_idx)))
    sample_nids = sampled_train_ids + sampled_val_ids + sampled_test_ids
    # sample_nids = sample_nids[:args.max_targets]
    subgraph_ids = node_ids_to_keep(graph, sample_nids, n_layers)
    print(f"len subgraph_ids = {len(subgraph_ids)}")
    sg = graph.subgraph(subgraph_ids, relabel_nodes=True)

    # sg.ndata['label'] = sg.ndata['label'].to(torch.int64)
    print(f"sg.ndata['label'] = {sg.ndata['label']}")
    return sg
    # dgl.save_graphs("{}_frac_{}_hops_{}_subgraph.bin".format(dataset, frac, n_layers), sg)

#############################################################################################

dataset = AsNodePredDataset(DglNodePropPredDataset(args.dataset, root=root))
# print(dataset.train_idx)
# print(dataset.val_idx)
# print(dataset.test_idx)

# print(type(dataset.train_idx))
# print(random.sample((dataset.train_idx).tolist(), 3)) 


# # # device = "cuda" if torch.cuda.is_available() else "cpu"
# # # device = "cpu"
graph = dataset[0]

isolated_nodes = ((graph.in_degrees() == 0) & (graph.out_degrees() == 0)).nonzero().squeeze(1)
print(f"isolated nodes = {isolated_nodes}, len(isolated_nodes) = {len(isolated_nodes)}")
# graph.remove_node(isolated_nodes)

print(f"g.ndata = {graph.ndata}")

# # train_ids = dataset.train_idx

# print(f"len(dataset.train_idx) = {len(dataset.train_idx)}")
# print(f"len(dataset.val_idx) = {len(dataset.val_idx)}")
# print(f"len(dataset.test_idx) = {len(dataset.test_idx)}")

# print(f"train_mask sum = {torch.sum(graph.ndata['train_mask'])}")
# print(f"val_mask sum = {torch.sum(graph.ndata['val_mask'])}")
# print(f"test_mask sum = {torch.sum(graph.ndata['test_mask'])}")

# valid_labels = torch.isfinite(graph.ndata['label']).nonzero().reshape(-1)
# print(f"valid_labels = {valid_labels}")
# print(f"len(valid_labels) = {len(valid_labels)}")


# train_ids = torch.nonzero(graph.ndata['train_mask']).reshape(-1)
# print(len(train_ids))
# valid_ids = torch.nonzero(graph.ndata['val_mask']).reshape(-1)
# print(len(valid_ids))
# test_ids = torch.nonzero(graph.ndata['test_mask']).reshape(-1)
# print(len(test_ids))


# frac = args.frac
# sampled_train_ids = random.sample((dataset.train_idx).tolist(), int(frac*len(dataset.train_idx)))
# print(f"sampled_train_ids = {len(sampled_train_ids)}")
# sampled_val_ids = random.sample((dataset.val_idx).tolist(), int(frac*len(dataset.val_idx)))
# print(f"len(sampled_val_ids) = {len(sampled_val_ids)}")
# sampled_test_ids = random.sample((dataset.test_idx).tolist(), int(frac*len(dataset.test_idx)))
# print(f"len(sampled_test_ids) = {len(sampled_test_ids)}")

# sample_nids = sampled_train_ids + sampled_val_ids + sampled_test_ids
# print(f"total sampled nids = {len(sample_nids)}")

# print(f"len(graph.ndata['train_mask']) = {len(graph.ndata['train_mask'])}")

# # print(f"len(graph.ndata['train_mask'][sample_nids]) {len(graph.ndata['train_mask'][sample_nids])}")



# subgraph_ids = node_ids_to_keep(graph, sample_nids, args.n_layers)

# print(f"len subgraph_ids = {len(subgraph_ids)}")


# # train_tiny_ids = node_ids_to_keep(graph, sampled_train_ids, 3)
# # valid_tiny_ids = node_ids_to_keep(graph, sampled_val_ids, 3)
# # test_tiny_ids = node_ids_to_keep(graph, sampled_test_ids, 3)

# # ids = torch.cat([train_tiny_ids, valid_tiny_ids, test_tiny_ids]).unique()

# # print(ids)

# # orgi_train_idx = orgi_train_idx[ids]
# # orgi_valid_idx = orgi_valid_idx[ids]
# # orgi_test_idx = orgi_test_idx[ids]


# sg = graph.subgraph(subgraph_ids, relabel_nodes=True)

# print(f"sg.ndata = {sg.ndata}")

# sg.ndata['label'] = sg.ndata['label'].to(torch.int64)

# print(f"labels = {len(sg.ndata['label'])}")
# print(f"max label = {max(sg.ndata['label'])}")

# print(f"len of sg train mask = {len(sg.ndata['train_mask'])}")


# func(dataset, 0.1, 3) # 13,719,273
# func(dataset, 0.01, 3) # 7,038,252
# func(dataset, 0.01, 2) # 1,737,962
# func(dataset, 0.1, 2) # 5,060,237
# func(dataset, 0.1, 4) # 27,640,781
# func(dataset, 0.01, 4) # 15,764,586
# func(dataset, 0.05, 3) # 10,687,657

def transform_labels(labels):
    mapping = {}
    sorted_uniques = torch.sort(torch.unique(labels[torch.logical_not(torch.isnan(labels))]))
    print(sorted_uniques[0])
    for i, el in enumerate(sorted_uniques[0]):
        labels[labels == el.item()] = i
    print(torch.sort(torch.unique(labels[torch.logical_not(torch.isnan(labels))]))[0])

# iterate = [(0.1, 2), (0.01, 3), (0.05, 3), (0.1, 3), (0.01, 4), (0.05, 4), (0.1, 4)]
# iterate = [(1.0, 2), (1.0, 3), (1.0, 4)]
iterate = [(args.frac, args.n_layers)]
for frac, layers in iterate:
    sg = func(dataset, frac, layers)

    # isolated_nodes = ((sg.in_degrees() == 0) & (sg.out_degrees() == 0)).nonzero().squeeze(1)
    # sg.remove_nodes(isolated_nodes)
    if args.dataset == "ogbn-arxiv" or args.dataset == 'ogbn-papers100M':
        sg.edata.clear()
        sg = dgl.to_bidirected(sg, copy_ndata=True)
        sg = dgl.remove_self_loop(sg)
        sg = dgl.add_self_loop(sg)
    else:
        sg.edata.clear()
        sg = dgl.remove_self_loop(sg)
        sg = dgl.add_self_loop(sg)


    transform_labels(sg.ndata['label'])
    sg.ndata['label'] = sg.ndata['label'].to(torch.int64)
    print(sg.ndata['label'])

    dgl.save_graphs("{}_frac_{}_hops_{}_subgraph.bin".format(args.dataset, frac*100, layers), sg)


    # dgl.save_graphs("{}_frac_{}_hops_{}_subgraph_cleared.bin".format(args.dataset, frac*100, layers), sg)

###################################################################################


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