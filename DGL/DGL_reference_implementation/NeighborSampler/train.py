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


import random
import wandb
wandb.login()

import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv, GraphConv
import tqdm
import sklearn.metrics
from parser import create_parser
from utils import load_data
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

import warnings
warnings.filterwarnings("ignore")

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



    def forward(self, mfgs, x):
        h_dst = x[:mfgs[0].num_dst_nodes()]  # <---
        h = self.layers[0](mfgs[0], (x, h_dst))
        for i in range(1, self.n_layers - 1):
            h_dst = h[:mfgs[i].num_dst_nodes()]  # <---
            h = self.layers[i](mfgs[i], (h, h_dst))
            # h = F.relu(h)
            h = self.activation(h)
            h = self.dropout(h)
        h_dst = h[:mfgs[-1].num_dst_nodes()]  # <---
        h = self.layers[-1](mfgs[-1], (h, h_dst))
        return h

    def inference(self, g, x):

        h = x
        for i in range(self.n_layers - 1):
            h = self.layers[i](g, h)
            h = self.activation(h)
            # h = self.dropout(h)
        h = self.layers[-1](g, h)
        return h

class GAT(nn.Module):
    def __init__(
        self, in_feats, n_hidden, n_classes, n_layers, num_heads
    ):
        super(GAT, self).__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.num_heads = num_heads

        self.layers = nn.ModuleList()
        self.layers.append(GATConv(in_feats, n_hidden, num_heads=num_heads))
        for _ in range(n_layers - 2):
            self.layers.append(GATConv(n_hidden*num_heads, n_hidden, num_heads=num_heads))
        self.layers.append(GATConv(n_hidden*num_heads, n_classes, num_heads=1))

    def forward(self, mfgs, x):
        # h_dst = x[:mfgs[0].num_dst_nodes()]  # <---
        # h = self.layers[0](mfgs[0], x)
        h = x
        for i in range(self.n_layers - 1):
            # h_dst = h[:mfgs[i].num_dst_nodes()]  # <---
            # print(mfgs[i], h.shape)
            h = self.layers[i](mfgs[i], h)
            # h = F.relu(h)
            # h = self.activation(h)
            # h = self.dropout(h)
            h = h.flatten(1)

        # h_dst = h[:mfgs[-1].num_dst_nodes()]  # <---
        h = self.layers[-1](mfgs[-1], h)
        # print(h.shape)
        h = h.mean(1)
        # print(h.shape)
        return h


class NSGCN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout=0.5):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        # two-layer GCN
        self.layers.append(GraphConv(in_feats, n_hidden, activation=F.relu))
        for i in range(1, n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=F.relu))

        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
    
    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h


def _get_data_loader(sampler, device, graph, nids, batch_size=1024):
    logger.info("Get train-val-test data loader")
    
    # print(dir(dataset))
    # print(dataset)
    # idx_split = dataset.get_idx_split()
    # train_nids = idx_split['train']
    # valid_nids = idx_split['valid']
    # test_nids = idx_split['test']

    train_nids, valid_nids, test_nids = nids
    logger.info("Get train data loader")
    train_dataloader = dgl.dataloading.DataLoader(
    # The following arguments are specific to DGL's DataLoader.
    graph,              # The graph
    train_nids,         # The node IDs to iterate over in minibatches
    sampler,            # The neighbor sampler
    device=device,      # Put the sampled MFGs on CPU or GPU
    # The following arguments are inherited from PyTorch DataLoader.
    batch_size=batch_size,    # Batch size
    shuffle=True,       # Whether to shuffle the nodes for every epoch
    drop_last=False,    # Whether to drop the last incomplete batch
    num_workers=0       # Number of sampler processes
    )
    logger.info("Get val data loader")
    valid_dataloader = dgl.dataloading.DataLoader(
    graph, valid_nids, sampler,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=0,
    device=device
    )

    logger.info("Get test data loader")
    test_dataloader = dgl.dataloading.DataLoader(
    graph, test_nids, sampler,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=0,
    device=device
    )

    logger.info("Train-val-test data loader created")
    
    return (train_dataloader, valid_dataloader, test_dataloader)

@torch.no_grad()
def evaluate2(logits, labels, mask):
    logits = logits[mask]
    labels = labels[mask]
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def train(graph, dataset, node_features, device, model, args):
    
    sampler = dgl.dataloading.NeighborSampler([args.fanout for _ in range(args.n_layers)])

    train_nids = np.where(graph.ndata['train_mask'])[0]
    valid_nids = np.where(graph.ndata['val_mask'])[0]
    test_nids = np.where(graph.ndata['test_mask'])[0]

    nids = (train_nids, valid_nids, test_nids)
    
    data = _get_data_loader(sampler, device, graph, nids, args.batch_size)

    train_dataloader, valid_dataloader, test_dataloader = data

    input_nodes, output_nodes, mfgs = example_minibatch = next(iter(train_dataloader))
    

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, eta_min=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.99, patience=20, min_lr=1e-4)

    best_train_acc = 0
    best_val_acc = 0
    best_test_acc = 0
    train_acc = 0
    val_acc = 0
    test_acc = 0
    # best_model_path = 'model.pt'
    # best_model = None
    train_time = 0
    for epoch in range(args.n_epochs):
        t0 = time.time()
        model.train()

        for step, (input_nodes, output_nodes, mfgs) in enumerate(train_dataloader):
            inputs = mfgs[0].srcdata['feat']
            labels = mfgs[-1].dstdata['label']
            # print(f"model  = {(model.parameters()).device}, mfgs = {mfgs.device}, inputs = {inputs.device}")
            predictions = model(mfgs, inputs)
            loss = F.cross_entropy(predictions, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        t1 = time.time()
        train_time += t1 - t0

        # scheduler.step(best_val_acc)
        # scheduler.step()
        if (epoch+1) % args.log_every == 0:
            model.eval()

            train_predictions = []
            train_labels = []
            val_predictions = []
            val_labels = []
            test_predictions = []
            test_labels = []
            with torch.no_grad():

                # Inference on entire graph
                # pred = model.inference(graph.to(device), graph.ndata['feat'].to(device))

                for input_nodes, output_nodes, mfgs in train_dataloader:
                    inputs = mfgs[0].srcdata['feat']
                    train_labels.append(mfgs[-1].dstdata['label'].cpu().numpy())
                    train_predictions.append(model(mfgs, inputs).argmax(1).cpu().numpy())
                train_predictions = np.concatenate(train_predictions)
                train_labels = np.concatenate(train_labels)
                train_acc = sklearn.metrics.accuracy_score(train_labels, train_predictions)

                # train_acc_fullgraph_no_sample = sklearn.metrics.accuracy_score(graph.ndata['label'][train_nids].cpu().numpy(), pred[train_nids].argmax(1).cpu().numpy())


                for input_nodes, output_nodes, mfgs in valid_dataloader:
                    inputs = mfgs[0].srcdata['feat']
                    val_labels.append(mfgs[-1].dstdata['label'].cpu().numpy())
                    val_predictions.append(model(mfgs, inputs).argmax(1).cpu().numpy())
                val_predictions = np.concatenate(val_predictions)
                val_labels = np.concatenate(val_labels)
                val_acc = sklearn.metrics.accuracy_score(val_labels, val_predictions)

                # val_acc_fullgraph_no_sample = sklearn.metrics.accuracy_score(graph.ndata['label'][valid_nids].cpu().numpy(), pred[valid_nids].argmax(1).cpu().numpy())

                for input_nodes, output_nodes, mfgs in test_dataloader:
                    inputs = mfgs[0].srcdata['feat']
                    test_labels.append(mfgs[-1].dstdata['label'].cpu().numpy())
                    test_predictions.append(model(mfgs, inputs).argmax(1).cpu().numpy())
                test_predictions = np.concatenate(test_predictions)
                test_labels = np.concatenate(test_labels)
                test_acc = sklearn.metrics.accuracy_score(test_labels, test_predictions)

                # test_acc_fullgraph_no_sample = sklearn.metrics.accuracy_score(graph.ndata['label'][test_nids].cpu().numpy(), pred[test_nids].argmax(1).cpu().numpy())
                t2 = time.time()

                if best_val_acc < val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    best_train_acc = train_acc

                logger.debug('Epoch {}, Train Acc {:.4f} (Best {:.4f}), Val Acc {:.4f} (Best {:.4f}), Test Acc {:.4f} (Best {:.4f})'.format(epoch, train_acc, best_train_acc, val_acc, best_val_acc, test_acc, best_test_acc))
                print(f"Train time = {t1-t0}, Eval time = {t2-t1}")
            wandb.log({'val_acc': val_acc,
                        'test_acc': test_acc,
                        'train_acc': train_acc,
                        'best_val_acc': best_val_acc,
                        'best_test_acc': best_test_acc,
                        'best_train_acc': best_train_acc,
                        
                        # 'train_acc_fullgraph_no_sample': train_acc_fullgraph_no_sample,
                        # 'val_acc_fullgraph_no_sample': val_acc_fullgraph_no_sample,
                        # 'test_acc_fullgraph_no_sample': test_acc_fullgraph_no_sample,

                        'lr': optimizer.param_groups[0]['lr'],
                        'train_time': train_time,
            })
            
    return best_train_acc, best_val_acc, best_test_acc, model


def main():
    args = create_parser()
        
    wandb.init(
        project="{}-SingleGPU-NS-{}".format(args.model, args.dataset),
        config={
            "n_epochs": args.n_epochs,
            "lr": args.lr,
            'weight_decay':args.weight_decay,
            "dropout": args.dropout, #random.uniform(0.5, 0.7),
            "n_hidden": args.n_hidden,
            "n_layers": args.n_layers,
            "agg": args.agg,
            "batch_size": args.batch_size,
            "fanout": args.fanout,
            "num_heads": args.num_heads,
            })


    config = wandb.config
    
    args.n_epochs = config.n_epochs
    args.n_layers = config.n_layers
    args.n_hidden = config.n_hidden
    args.dropout = config.dropout
    args.batch_size = config.batch_size
    args.fanout = config.fanout
    args.num_heads = config.num_heads
    args.lr = config.lr
    args.weight_decay = config.weight_decay
    args.agg = config.agg
    
    if args.seed:
        torch.manual_seed(args.seed)

    wandb.log({
        'seed': torch.initial_seed() & ((1<<63)-1),
    })
    dataset = load_data(args.dataset)
    graph = dataset[
        0
    ]  # already prepares ndata['label'/'train_mask'/'val_mask'/'test_mask']

    if args.dataset == "ogbn-arxiv":
        graph.edata.clear()
        graph = dgl.to_bidirected(graph, copy_ndata=True)
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)
    else:
        graph.edata.clear()
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)

    device = "cuda:{}".format(args.device_id) if torch.cuda.is_available() else "cpu"

    node_features = graph.ndata['feat']
    in_feats = node_features.shape[1]
    n_classes = dataset.num_classes
    activation = F.relu

    if 'sage' in args.model.lower():
        print("using graphsage model")
        model = SAGE(in_feats, args.n_hidden, n_classes, args.n_layers, args.dropout, activation, aggregator_type=args.agg).to(device)
    elif 'gat' in args.model.lower():
        model = GAT(in_feats, args.n_hidden, n_classes, args.n_layers, args.num_heads).to(device)

    elif 'gcn' in args.model.lower():
        model = NSGCN(in_feats, args.n_hidden, n_classes, args.n_layers, activation, args.dropout).to(device)
    train(graph, dataset, node_features, device, model, args)




if __name__ == "__main__":
    main()

    # args = create_parser()
    # sweep_configuration = {
    #     'name': 'HPO-HPO',
    #     'method': 'grid',
    #     'metric': {'goal': 'maximize', 'name': 'val_acc'},
    #     'parameters': 
    #     {
    #         # 'lr': {'distribution': 'uniform', 'min': 5*1e-4, 'max': 1e-2},
    #         # 'n_hidden': {'distribution': 'int_uniform', 'min': 64, 'max': 1024},
    #         # 'n_hidden': {'values': [256]},
    #         'n_layers': {'values': [2,3,4,5]},
    #         # 'n_layers': {'distribution': 'int_uniform', 'min': 2, 'max': 10},
    #         # 'dropout': {'distribution': 'uniform', 'min': 0.2, 'max': 0.8},
    #         # "agg": {'values': ["mean", "gcn", "pool"]},
    #         # 'num_epochs': {'values': [2000, 4000, 6000, 8000]},
    #         # 'batch_size': {'values': [128, 256, 512, 1024]},
    #         'fanout': {'values': [8, 10, 15]},
    #         # 'fanout': {'distribution': 'int_uniform', 'min': 3, 'max': 10},
    #         # 'num_heads': {'distribution': 'int_uniform', 'min': 1, 'max': 10},
    #         'num_heads': {'values': [4, 6, 8]},
            
    #     }
    # }
    # sweep_id = wandb.sweep(sweep=sweep_configuration, project="{}-SingleGPU-NS-{}".format(args.model, args.dataset))

    # wandb.agent(sweep_id, function=main, count=5000)

#tmux
# ctrl+b -> d
# attach -t
# tmux attach -t 0