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
from dgl.nn import SAGEConv
import tqdm
import sklearn.metrics

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

import warnings
warnings.filterwarnings("ignore")

class Model(nn.Module):
    def __init__(
        self, in_feats, n_hidden, n_classes, n_layers, dropout, activation, aggregator_type='mean'
    ):
        super(Model, self).__init__()
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


def train():
    
    wandb.init(
        project="pubmed_NS_sage",
        config={
            "num_epochs": 1000,
            "lr": 1e-3,
            'weight_decay':5e-4,
            "dropout": 0.6,#random.uniform(0.5, 0.7),
            "n_hidden": 512,
            "n_layers": 3,
            "agg": "mean",
            "batch_size": 1024,
            "fanout": 16,
            })


    config = wandb.config
    
    n_layers = config.n_layers
    n_hidden = config.n_hidden
    num_epochs = config.num_epochs
    dropout = config.dropout
    batch_size = config.batch_size
    fanout = config.fanout
    lr = config.lr
    weight_decay=config.weight_decay
    agg = config.agg
    
    root="../dataset/"
    # dataset = DglNodePropPredDataset('ogbn-arxiv', root=root)
    # torch.cuda.set_device(3)

    dataset = dgl.data.PubmedGraphDataset(raw_dir=root)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sampler = dgl.dataloading.NeighborSampler([fanout for _ in range(n_layers)])
    
    graph = dataset[0]
    # node_labels = graph.ndata['label']
    graph = dgl.add_reverse_edges(graph)
    # graph.ndata['label'] = node_labels[:, 0]

    train_nids = np.where(graph.ndata['train_mask'])[0]
    valid_nids = np.where(graph.ndata['val_mask'])[0]
    test_nids = np.where(graph.ndata['test_mask'])[0]

    node_features = graph.ndata['feat']
    in_feats = node_features.shape[1]
    n_classes = dataset.num_classes

    nids = (train_nids, valid_nids, test_nids)
    
    data = _get_data_loader(sampler, device, graph, nids, batch_size)

    train_dataloader, valid_dataloader, test_dataloader = data

    input_nodes, output_nodes, mfgs = example_minibatch = next(iter(train_dataloader))
    activation = F.relu

    model = Model(in_feats, n_hidden, n_classes, n_layers, dropout, activation, aggregator_type=agg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, eta_min=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.99, patience=20, min_lr=1e-5)

    best_train_acc = 0
    best_val_acc = 0
    best_test_acc = 0

    best_model_path = 'model.pt'
    best_model = None

    for epoch in range(num_epochs):
        model.train()
        
        

        for step, (input_nodes, output_nodes, mfgs) in enumerate(train_dataloader):
            inputs = mfgs[0].srcdata['feat']
            labels = mfgs[-1].dstdata['label']
            predictions = model(mfgs, inputs)
            loss = F.cross_entropy(predictions, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step(best_val_acc)

        if epoch % 5 == 0:
            model.eval()

            train_predictions = []
            train_labels = []
            val_predictions = []
            val_labels = []
            test_predictions = []
            test_labels = []
            with torch.no_grad():

                # Inference on entire graph
                pred = model.inference(graph.to(device), graph.ndata['feat'].to(device))

                for input_nodes, output_nodes, mfgs in train_dataloader:
                    inputs = mfgs[0].srcdata['feat']
                    train_labels.append(mfgs[-1].dstdata['label'].cpu().numpy())
                    train_predictions.append(model(mfgs, inputs).argmax(1).cpu().numpy())
                train_predictions = np.concatenate(train_predictions)
                train_labels = np.concatenate(train_labels)
                train_acc = sklearn.metrics.accuracy_score(train_labels, train_predictions)

                train_acc_fullgraph_no_sample = sklearn.metrics.accuracy_score(graph.ndata['label'][train_nids].cpu().numpy(), pred[train_nids].argmax(1).cpu().numpy())


                for input_nodes, output_nodes, mfgs in valid_dataloader:
                    inputs = mfgs[0].srcdata['feat']
                    val_labels.append(mfgs[-1].dstdata['label'].cpu().numpy())
                    val_predictions.append(model(mfgs, inputs).argmax(1).cpu().numpy())
                val_predictions = np.concatenate(val_predictions)
                val_labels = np.concatenate(val_labels)
                val_acc = sklearn.metrics.accuracy_score(val_labels, val_predictions)

                val_acc_fullgraph_no_sample = sklearn.metrics.accuracy_score(graph.ndata['label'][valid_nids].cpu().numpy(), pred[valid_nids].argmax(1).cpu().numpy())

                for input_nodes, output_nodes, mfgs in test_dataloader:
                    inputs = mfgs[0].srcdata['feat']
                    test_labels.append(mfgs[-1].dstdata['label'].cpu().numpy())
                    test_predictions.append(model(mfgs, inputs).argmax(1).cpu().numpy())
                test_predictions = np.concatenate(test_predictions)
                test_labels = np.concatenate(test_labels)
                test_acc = sklearn.metrics.accuracy_score(test_labels, test_predictions)

                test_acc_fullgraph_no_sample = sklearn.metrics.accuracy_score(graph.ndata['label'][test_nids].cpu().numpy(), pred[test_nids].argmax(1).cpu().numpy())

                if best_val_acc < val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    best_train_acc = train_acc

                logger.debug('Epoch {}, Train Acc {:.4f} (Best {:.4f}), Val Acc {:.4f} (Best {:.4f}), Test Acc {:.4f} (Best {:.4f})'.format(epoch, train_acc, best_train_acc, val_acc, best_val_acc, test_acc, best_test_acc))
            
            wandb.log({'val_acc': val_acc,
                        'test_acc': test_acc,
                        'train_acc': train_acc,
                        'best_val_acc': best_val_acc,
                        'best_test_acc': best_test_acc,
                        'best_train_acc': best_train_acc,
                        
                        'train_acc_fullgraph_no_sample': train_acc_fullgraph_no_sample,
                        'val_acc_fullgraph_no_sample': val_acc_fullgraph_no_sample,
                        'test_acc_fullgraph_no_sample': test_acc_fullgraph_no_sample,

                        'lr': optimizer.param_groups[0]['lr'],
            })
            
    return best_train_acc, best_val_acc, best_test_acc, model

if __name__ == "__main__":
    
    # args = parse_args_fn()
    train_val, val_acc, test_acc, model = train()
    # runs = 2
    # avg_train_acc = avg_val_acc = avg_test_acc = 0
    # for _ in range(runs):
    #     train_acc, val_acc, test_acc, model = train()
    #     avg_train_acc += train_acc
    #     avg_val_acc += val_acc
    #     avg_test_acc += test_acc
    # print(avg_train_acc/runs, avg_val_acc/runs, avg_test_acc/runs)

    # sweep_configuration = {
    #     'method': 'random',
    #     'metric': {'goal': 'maximize', 'name': 'val_acc'},
    #     'parameters': 
    #     {
    #         # 'lr': {'distribution': 'log_uniform_values', 'min': 5*1e-3, 'max': 1e-1},
    #         'n_hidden': {'distribution': 'int_uniform', 'min': 64, 'max': 1024},
    #         # 'n_layers': {'distribution': 'int_uniform', 'min': 3, 'max': 10},
    #         # 'dropout': {'distribution': 'uniform', 'min': 0.2, 'max': 0.8},
    #         # "agg": {'values': ["mean", "gcn", "pool"]},
    #         # 'num_epochs': {'values': [2000, 4000, 6000, 8000]},
    #         # 'batch_size': {'values': [128, 256, 512, 1024]},
    #         # 'fanout': {'distribution': 'int_uniform', 'min': 3, 'max': 25},
    #     }
    # }
    # sweep_id = wandb.sweep(sweep=sweep_configuration, project='pubmed_NS_sage')

    # wandb.agent(sweep_id, function=train, count=30)

#tmux
# ctrl+b -> d
# attach -t
# tmux attach -t 0