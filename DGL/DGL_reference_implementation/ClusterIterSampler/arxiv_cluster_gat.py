import argparse
import json
import logging
import os
import sys
import pickle

import dgl
import torch
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
import time 
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau


import random
import wandb
wandb.login()

import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATConv
import tqdm
import sklearn.metrics

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

import warnings
warnings.filterwarnings("ignore")


class GAT(nn.Module):
    def __init__(
        self, in_feats, num_heads, n_hidden, n_classes, n_layers
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


def _get_data_loader(graph, num_parts, sampler, device, shuffle=True, batch_size=1024):
    logger.info("Get data loader")
    
    dataloader = dgl.dataloading.DataLoader(
    # The following arguments are specific to DGL's DataLoader.
    graph,              # The graph
    torch.arange(num_parts),         # The node IDs to iterate over in minibatches
    sampler,            # The neighbor sampler
    device=device,      # Put the sampled MFGs on CPU or GPU
    # The following arguments are inherited from PyTorch DataLoader.
    batch_size=batch_size,    # Batch size
    shuffle=shuffle,       # Whether to shuffle the nodes for every epoch
    drop_last=False,    # Whether to drop the last incomplete batch
    num_workers=0       # Number of sampler processes
    )

    logger.info("Data loader created")
    
    return dataloader


@torch.no_grad()
def evaluate(evaluator, predictions, labels):
    acc = evaluator.eval({
        'y_true': torch.reshape(labels, (-1, 1)),
        'y_pred': torch.reshape(predictions, (-1, 1)),
    })['acc']
    # eacc = sklearn.metrics.accuracy_score(labels, predictions)
    return acc

def train():
    
    wandb.init(
        project="mini-batch-cluster-arxiv-gat",
        config={
            "num_epochs": 10,
            "lr": 2.5*1e-4,
            "dropout": random.uniform(0.0, 0.5),
            "n_hidden": 256,
            "n_layers": 3,
            "num_heads":2,
            "agg": "mean",
            "batch_size": 256,
            "num_parts": 8000,
            })


    config = wandb.config
    
    n_layers = config.n_layers
    n_hidden = config.n_hidden
    num_epochs = config.num_epochs
    dropout = config.dropout
    batch_size = config.batch_size
    lr = config.lr
    agg = config.agg
    num_parts = config.num_parts
    num_heads = config.num_heads

    
    root="../dataset/"
    dataset = DglNodePropPredDataset('ogbn-arxiv', root=root)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    evaluator = Evaluator(name='ogbn-arxiv')

    idx_split = dataset.get_idx_split()
    train_nids = idx_split['train']
    valid_nids = idx_split['valid']
    test_nids = idx_split['test']

    graph, node_labels = dataset[0]


    graph = dgl.add_reverse_edges(graph)
    graph.ndata['label'] = node_labels[:, 0]
    
    node_features = graph.ndata['feat']
    in_feats = node_features.shape[1]
    n_classes = (node_labels.max() + 1).item()

    # sampler = dgl.dataloading.NeighborSampler([fanout for _ in range(n_layers)])
    # sampler = dgl.dataloading.SAINTSampler(mode='node', budget=budget)
    train_sampler = dgl.dataloading.ClusterGCNSampler(graph.subgraph(train_nids), num_parts, cache_path='train_cluster_gcn_{}_{}_{}_{}.pkl'.format(n_layers, n_hidden, lr, num_parts))
    valid_sampler = dgl.dataloading.ClusterGCNSampler(graph.subgraph(valid_nids), num_parts, cache_path='valid_cluster_gcn_{}_{}_{}_{}.pkl'.format(n_layers, n_hidden, lr, num_parts))
    test_sampler = dgl.dataloading.ClusterGCNSampler(graph.subgraph(test_nids), num_parts, cache_path='test_cluster_gcn_{}_{}_{}_{}.pkl'.format(n_layers, n_hidden, lr, num_parts))

    train_dataloader = _get_data_loader(graph.subgraph(train_nids), num_parts, train_sampler, device, shuffle=True, batch_size=batch_size)
    valid_dataloader = _get_data_loader(graph.subgraph(valid_nids), num_parts, valid_sampler, device, shuffle=False, batch_size=batch_size)
    test_dataloader = _get_data_loader(graph.subgraph(test_nids), num_parts, test_sampler, device, shuffle=False, batch_size=batch_size)

    activation = F.relu

    model = GAT(in_feats, num_heads, n_hidden, n_classes, n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, eta_min=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', cooldown=10, factor=0.95, patience=30, min_lr=1e-4)

    best_train_acc = 0
    best_eval_acc = 0
    best_test_acc = 0

    best_model_path = 'model.pt'
    best_model = None
    total_time = 0

    time_load = 0
    time_forward = 0
    time_backward = 0
    total_time = 0
    for epoch in range(num_epochs):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.train()
        tic = time.time()

        for step, subg in enumerate(train_dataloader):
            tic_start = time.time()
            inputs = subg.ndata['feat']
            labels = subg.ndata['label']
            tic_step = time.time()
            predictions = model(subg, inputs)
            loss = F.cross_entropy(predictions, labels)
            optimizer.zero_grad()
            tic_forward = time.time()
            loss.backward()
            optimizer.step()
            tic_backward = time.time()

            time_load += tic_step - tic_start
            time_forward += tic_forward - tic_step
            time_backward += tic_backward - tic_forward

        scheduler.step(best_eval_acc)
        toc = time.time()
        total_time += toc - tic

        if epoch % 5 == 0:
            model.eval()
            train_predictions = []
            train_labels = []
            val_predictions = []
            val_labels = []
            test_predictions = []
            test_labels = []

            with torch.no_grad():
                for subg in train_dataloader:
                    inputs = subg.ndata['feat']
                    train_labels.append(subg.ndata['label'].cpu().numpy())
                    train_predictions.append(model(subg, inputs).argmax(1).cpu().numpy())
                train_predictions = np.concatenate(train_predictions)
                train_labels = np.concatenate(train_labels)
                train_acc = sklearn.metrics.accuracy_score(train_labels, train_predictions)
                
                for subg in valid_dataloader:
                    inputs = subg.ndata['feat']
                    val_labels.append(subg.ndata['label'].cpu().numpy())
                    val_predictions.append(model(subg, inputs).argmax(1).cpu().numpy())
                val_predictions = np.concatenate(val_predictions)
                val_labels = np.concatenate(val_labels)
                eval_acc = sklearn.metrics.accuracy_score(val_labels, val_predictions)

                for subg in test_dataloader:
                    inputs = subg.ndata['feat']
                    test_labels.append(subg.ndata['label'].cpu().numpy())
                    test_predictions.append(model(subg, inputs).argmax(1).cpu().numpy())
                test_predictions = np.concatenate(test_predictions)
                test_labels = np.concatenate(test_labels)
                test_acc = sklearn.metrics.accuracy_score(test_labels, test_predictions)
                
                device = "cpu"
                model = model.to(device)
                pred = model(graph.to(device), graph.ndata['feat'].to(device))
                train_acc_fullgraph_no_sample = evaluate(evaluator, pred[train_nids].argmax(1), graph.ndata['label'][train_nids].to(device))
                val_acc_fullgraph_no_sample = evaluate(evaluator, pred[valid_nids].argmax(1), graph.ndata['label'][valid_nids].to(device))
                test_acc_fullgraph_no_sample = evaluate(evaluator, pred[test_nids].argmax(1), graph.ndata['label'][test_nids].to(device))

                if best_eval_acc < eval_acc:
                    best_eval_acc = eval_acc
                    best_model = model
                    best_test_acc = test_acc
                    best_train_acc = train_acc
                logger.debug('Epoch {}, Train Acc {:.4f} (Best {:.4f}), Val Acc {:.4f} (Best {:.4f}), Test Acc {:.4f} (Best {:.4f})'.format(epoch, train_acc, best_train_acc, eval_acc, best_eval_acc, test_acc, best_test_acc))
            
            wandb.log({'val_acc': eval_acc,
                        'test_acc': test_acc,
                        'train_acc': train_acc,
                        'best_eval_acc': best_eval_acc,
                        'best_test_acc': best_test_acc,
                        'best_train_acc': best_train_acc,
                        'train_acc_fullgraph_no_sample': train_acc_fullgraph_no_sample,
                        'val_acc_fullgraph_no_sample': val_acc_fullgraph_no_sample,
                        'test_acc_fullgraph_no_sample': test_acc_fullgraph_no_sample,
                        # 'lr': scheduler.get_last_lr()[0],
                        'lr': optimizer.param_groups[0]['lr'],
            })
            
    logger.debug("total time for {} epochs = {}".format(num_epochs, total_time))
    logger.debug("avg time per epoch = {}".format(total_time/num_epochs))
    return best_eval_acc, model

if __name__ == "__main__":
    
    # args = parse_args_fn()

    eval_acc, model = train()
        
    
    # sweep_configuration = {
    #     'method': 'grid',
    #     'metric': {'goal': 'maximize', 'name': 'val_acc'},
    #     'parameters': 
    #     {
    #         # 'n_hidden': {'distribution': 'int_uniform', 'min': 128, 'max': 2048},
    #         # 'n_layers': {'distribution': 'int_uniform', 'min': 3, 'max': 20},
    #         # 'lr': {'distribution': 'uniform', 'max': 2e-3, 'min': 1e-4},
    #         # 'dropout': {'distribution': 'uniform', 'min': 0.5, 'max': 0.8},
    #         # "agg": {'values': ["mean", "gcn", "pool"]},
    #         # 'num_epochs': {'values': [2000, 4000, 6000, 8000]},
    #         'batch_size': {'values': [256, 512, 1024]},
    #         # 'num_parts': {'distribution': 'int_uniform', 'min': 100, 'max': 10000},
    #     }
    # }
    # sweep_id = wandb.sweep(sweep=sweep_configuration, project='mini-batch-cluster-arxiv')

    # wandb.agent(sweep_id, function=train, count=3)

#tmux
# ctrl+b -> d
# attach -t
# tmux attach -t 0