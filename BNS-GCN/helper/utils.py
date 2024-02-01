import os

import scipy
import dgl
from dgl.data import RedditDataset, YelpDataset
from dgl.distributed import partition_graph
from helper.context import *
from ogb.nodeproppred import DglNodePropPredDataset
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
import json

import torch
from dgl.data import RedditDataset
import torch.distributed as dist
import time

from dgl.data import DGLDataset
import pandas as pd

class TransferTag:
    NODE = 0
    FEAT = 1
    DEG = 2


def load_ogb_dataset(name, data_path):
    dataset = DglNodePropPredDataset(name=name, root='/home/ubuntu/gnn_mini_vs_full/GNN_minibatch_vs_fullbatch/dataset')
    split_idx = dataset.get_idx_split()
    g, label = dataset[0]
    n_node = g.num_nodes()
    node_data = g.ndata
    node_data['label'] = label.view(-1).long()
    node_data['train_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['val_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['test_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['train_mask'][split_idx["train"]] = True
    node_data['val_mask'][split_idx["valid"]] = True
    node_data['test_mask'][split_idx["test"]] = True

    return g


class OrkutDataset(DGLDataset):
    def __init__(self):
        super().__init__(name="orkut")

    def process(self):
        root = "/work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/dataset"
        edges_data = pd.read_csv(root + "/orkut/orkut_edges.csv")
        node_labels = pd.read_csv(root + "/orkut/orkut_labels.csv")


        node_features = torch.load(root + '/orkut/orkut_features.pt')
        # print(f"node_features = {node_features}")

        node_labels = torch.from_numpy(
            node_labels.astype("category").to_numpy()
        ).view(-1)
        # print(f"node_labels = {node_labels}")

        self.num_classes = (node_labels.max() + 1).item()
        # edge_features = torch.from_numpy(edges_data["Weight"].to_numpy())
        edges_src = torch.from_numpy(edges_data["Src"].to_numpy())
        edges_dst = torch.from_numpy(edges_data["Dst"].to_numpy())
        # print(f"node_features.shape = {node_features.shape}")
        self.graph = dgl.graph(
            (edges_src, edges_dst), num_nodes=node_features.shape[0]
        )
        self.graph.ndata["feat"] = node_features
        self.graph.ndata["label"] = node_labels
        # self.graph.edata["weight"] = edge_features

        # If your dataset is a node classification dataset, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.
        n_nodes = node_features.shape[0]
        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask[n_train : n_train + n_val] = True
        test_mask[n_train + n_val :] = True
        self.graph.ndata["train_mask"] = train_mask
        self.graph.ndata["val_mask"] = val_mask
        self.graph.ndata["test_mask"] = test_mask

        self.train_idx = self.graph.ndata["train_mask"].nonzero().view(-1)
        self.val_idx = self.graph.ndata["val_mask"].nonzero().view(-1)
        self.test_idx = self.graph.ndata["test_mask"].nonzero().view(-1)


    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1


def load_orkut():
    dataset = OrkutDataset()
    g = dataset[0]
    g = dgl.to_bidirected(g, copy_ndata=True)
    g.ndata['in_deg'] = g.in_degrees()
    g.ndata['out_deg'] = g.out_degrees()
    # g = dgl.remove_self_loop(g)
    # g = dgl.add_self_loop(g)
    return g



def load_subgraph(dataset_path):
    g, _ = dgl.load_graphs(dataset_path)
    # print(g)
    g = g[0]
    # g.ndata['label'] = g.ndata['label'].to(torch.int64)
    n_feat = g.ndata['feat'].shape[1]
    print("train_mask shape = {}".format(g.ndata['train_mask'].shape))
    print("label shape = {}".format(g.ndata['label'].shape))
    
    if g.ndata['label'].dim() == 1:
    
        n_class = int(torch.max(torch.unique(g.ndata['label'][torch.logical_not(torch.isnan(g.ndata['label']))])).item()) + 1 # g.ndata['label'].max().item() + 1
    else:
        n_class = g.ndata['label'].shape[1]
    return g, n_feat, n_class


def load_pubmed():
    root = "/home/ubuntu/gnn_mini_vs_full/GNN_minibatch_vs_fullbatch/dataset"
    dataset = dgl.data.PubmedGraphDataset(raw_dir=root)
    g = dataset[0]
    return g

def load_data(args):
    if args.dataset == 'reddit':
        data = RedditDataset(raw_dir='/home/ubuntu/gnn_mini_vs_full/GNN_minibatch_vs_fullbatch/dataset')
        g = data[0]
    elif args.dataset == 'ogbn-products':
        g = load_ogb_dataset('ogbn-products', '/home/ubuntu/gnn_mini_vs_full/GNN_minibatch_vs_fullbatch/dataset')
    elif args.dataset == 'ogbn-arxiv':
        g = load_ogb_dataset('ogbn-arxiv', '/home/ubuntu/gnn_mini_vs_full/GNN_minibatch_vs_fullbatch/dataset')

    elif args.dataset == 'ogbn-papers100m':
        # g = load_ogb_dataset('ogbn-papers100M', '/home/ubuntu/gnn_mini_vs_full/GNN_minibatch_vs_fullbatch/dataset')
        return load_subgraph(args.dataset_subgraph_path)
    elif args.dataset == 'pubmed':
        g = load_pubmed()
    elif args.dataset == 'yelp':
        data = YelpDataset(raw_dir='/home/ubuntu/gnn_mini_vs_full/GNN_minibatch_vs_fullbatch/dataset')
        g = data[0]
        g.ndata['label'] = g.ndata['label'].float()
        # TODO: remove the following three lines later (see Issue #4806 of DGL).
        g.ndata['train_mask'] = g.ndata['train_mask'].bool()
        g.ndata['val_mask'] = g.ndata['val_mask'].bool()
        g.ndata['test_mask'] = g.ndata['test_mask'].bool()
        feats = g.ndata['feat']
        scaler = StandardScaler()
        scaler.fit(feats[g.ndata['train_mask']])
        feats = scaler.transform(feats)
        g.ndata['feat'] = torch.tensor(feats, dtype=torch.float)
    elif args.dataset == "orkut":
        g = load_orkut()
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    n_feat = g.ndata['feat'].shape[1]
    if g.ndata['label'].dim() == 1:
        n_class = g.ndata['label'].max().item() + 1
    else:
        n_class = g.ndata['label'].shape[1]

    # g.edata.clear()
    # g = dgl.remove_self_loop(g)
    # g = dgl.add_self_loop(g)

    if args.dataset == "ogbn-arxiv":
        g.edata.clear()
        g = dgl.to_bidirected(g, copy_ndata=True)
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
    else:
        g.edata.clear()
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
    return g, n_feat, n_class


def graph_partition(args):

    g, n_feat, n_class = load_data(args)
    if args.inductive:
        g = g.subgraph(g.ndata['train_mask'])

    n_class = n_class
    n_feat = n_feat
    n_train = g.ndata['train_mask'].int().sum().item()

    graph_dir = os.path.join(args.part_path, args.graph_name)
    part_config = os.path.join(graph_dir, args.graph_name + '.json')

    # TODO: after being saved, a bool tensor becomes a uint8 tensor (including 'inner_node')
    if not os.path.exists(part_config):
        with g.local_scope():
            if args.inductive:
                g.ndata.pop('val_mask')
                g.ndata.pop('test_mask')
            g.ndata['in_deg'] = g.in_degrees()
            g.ndata['out_deg'] = g.out_degrees()
            partition_graph(g, args.graph_name, args.n_partitions, graph_dir,  part_method=args.partition_method,
                            balance_edges=False, objtype=args.partition_obj)
    
    with open(os.path.join(graph_dir, 'meta.json'), 'w') as f:
        json.dump({'n_feat': n_feat, 'n_class': n_class, 'n_train': n_train}, f)


def load_partition(args, rank):

    graph_dir = os.path.join(args.part_path, args.graph_name)
    part_config = os.path.join(graph_dir, args.graph_name + '.json')

    print('loading partitions')

    subg, node_feat, _, gpb, _, node_type, _ = dgl.distributed.load_partition(part_config, rank)
    node_type = node_type[0]
    node_feat[dgl.NID] = subg.ndata[dgl.NID]
