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

    def forward(self, g, x):
        h = x
        for l, conv in enumerate(self.layers):
            h = conv(g, h)
            # print("self.activation = {}".format(type(self.activation)))
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

def _get_data_loader(sampler, device, graph, nids, batch_size=1024):
    logger.info("Get train-val-test data loader")
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
    batch_size=1e6,#batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=0,
    device=device
    )

    logger.info("Get test data loader")
    test_dataloader = dgl.dataloading.DataLoader(
    graph, test_nids, sampler,
    batch_size=1e6,#batch_size
    shuffle=False,
    drop_last=False,
    num_workers=0,
    device=device
    )

    logger.info("Train-val-test data loader created")
    
    return (train_dataloader, valid_dataloader, test_dataloader)

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
        project="debugging",
        config={
            "num_epochs": 1000,
            "lr": 1e-3,
            "dropout": random.uniform(0.3, 0.6),
            "n_hidden": 1024,
            "n_layers": 3,
            "agg": "mean",
            "batch_size": 2**10,
            "budget": 5000,
            })


    config = wandb.config
    n_layers = config.n_layers
    n_hidden = config.n_hidden
    num_epochs = config.num_epochs
    dropout = config.dropout
    batch_size = config.batch_size
    lr = config.lr
    agg = config.agg
    budget = config.budget
    activation = F.relu
    
    # loading the dataset
    root="../dataset/"
    dataset = DglNodePropPredDataset('ogbn-arxiv', root=root)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # getting train, test, val splits
    idx_split = dataset.get_idx_split()
    train_nids = idx_split['train']
    valid_nids = idx_split['valid']
    test_nids = idx_split['test']

    # general pre-processing of the graph
    graph, node_labels = dataset[0]
    graph = dgl.add_reverse_edges(graph)
    graph.ndata['label'] = node_labels[:, 0]

    node_features = graph.ndata['feat']
    in_feats = node_features.shape[1]
    n_classes = (node_labels.max() + 1).item()

    # creating the sampler

    # sampler = dgl.dataloading.NeighborSampler([fanout for _ in range(n_layers)])
    sampler = dgl.dataloading.SAINTSampler(mode='walk', budget=(256, 512))

    data = _get_data_loader(sampler, device, graph, (train_nids, valid_nids, test_nids), batch_size)

    train_dataloader, valid_dataloader, test_dataloader = data

    model = Model(in_feats, n_hidden, n_classes, n_layers, dropout, activation, aggregator_type=agg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler1 = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, eta_min=1e-3)
    scheduler2 = ReduceLROnPlateau(optimizer, mode='max', factor=0.99, patience=20, min_lr=1e-5)

    evaluator = Evaluator(name='ogbn-arxiv')
    best_train_acc = 0
    best_val_acc = 0
    best_test_acc = 0

    best_model = None

    for epoch in range(num_epochs):
        model.train()

        # Training
        for step, subg in enumerate(train_dataloader):
            inputs = subg.ndata['feat']
            labels = subg.ndata['label']
            predictions = model(subg, inputs)
            loss = F.cross_entropy(predictions, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        scheduler2.step(best_val_acc)
        # scheduler1.step()

        # Evaluating
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
                pred = model(graph.to(device), graph.ndata['feat'].to(device))

                # Inference on sampled subgraph
                for subg in train_dataloader:
                    inputs = subg.ndata['feat']
                    train_labels.append(subg.ndata['label'])
                    train_predictions.append(model(subg, inputs).argmax(1))
                train_predictions = torch.cat(train_predictions)
                train_labels = torch.cat(train_labels)
                train_acc_subgraph_sample = sklearn.metrics.accuracy_score(train_labels.cpu().numpy(), train_predictions.cpu().numpy())

                train_acc_fullgraph_no_sample = evaluate(evaluator, pred[train_nids].argmax(1), graph.ndata['label'][train_nids])
                
                # Inference on subgraph
                pred_train = model(graph.subgraph(train_nids).to(device), graph.ndata['feat'][train_nids].to(device))
                train_acc_subgraph_no_sample = evaluate(evaluator, pred_train.argmax(1), graph.ndata['label'][train_nids])
                
                # Inference on sampled subgraph
                for subg in valid_dataloader:
                    inputs = subg.ndata['feat']
                    val_labels.append(subg.ndata['label'])
                    val_predictions.append(model(subg, inputs).argmax(1))
                val_predictions = torch.cat(val_predictions)
                val_labels = torch.cat(val_labels)
                val_acc_subgraph_sample = sklearn.metrics.accuracy_score(val_labels.cpu().numpy(), val_predictions.cpu().numpy())
                
                val_acc_fullgraph_no_sample = evaluate(evaluator, pred[valid_nids].argmax(1), graph.ndata['label'][valid_nids])

                # Inference on subgraph
                pred_valid = model(graph.subgraph(valid_nids).to(device), graph.ndata['feat'][valid_nids].to(device))
                val_acc_subgraph_no_sample = evaluate(evaluator, pred_valid.argmax(1), graph.ndata['label'][valid_nids])

                # Inference on sampled subgraph
                for subg in test_dataloader:
                    inputs = subg.ndata['feat']
                    test_labels.append(subg.ndata['label'])
                    test_predictions.append(model(subg, inputs).argmax(1))
                test_predictions = torch.cat(test_predictions)
                test_labels = torch.cat(test_labels)
                test_acc_subgraph_sample = sklearn.metrics.accuracy_score(test_labels.cpu().numpy(), test_predictions.cpu().numpy())
                
                test_acc_fullgraph_no_sample = evaluate(evaluator, pred[test_nids].argmax(1), graph.ndata['label'][test_nids])

                # Inference on subgraph
                pred_test = model(graph.subgraph(test_nids).to(device), graph.ndata['feat'][test_nids].to(device))
                test_acc_subgraph_no_sample = evaluate(evaluator, pred_test.argmax(1), graph.ndata['label'][test_nids])
                
                # Storing the best accuracies
                if best_val_acc < val_acc_subgraph_sample:
                    best_val_acc = val_acc_subgraph_sample
                    best_model = model
                    best_test_acc = test_acc_subgraph_sample
                    best_train_acc = train_acc_subgraph_sample
                logger.debug('Epoch {}, Train Acc {:.4f} (Best {:.4f}), Val Acc {:.4f} (Best {:.4f}), Test Acc {:.4f} (Best {:.4f})'.format(epoch, train_acc_subgraph_sample, best_train_acc, val_acc_subgraph_sample, best_val_acc, test_acc_subgraph_sample, best_test_acc))
            
            wandb.log({'val_acc': val_acc_subgraph_sample,
                        'test_acc': test_acc_subgraph_sample,
                        'train_acc': train_acc_subgraph_sample,
                        'best_val_acc': best_val_acc,
                        'best_test_acc': best_test_acc,
                        'best_train_acc': best_train_acc,
                        # 'lr': scheduler.get_last_lr()[0],
                        'lr': optimizer.param_groups[0]['lr'],

                        'train_diff': train_acc_fullgraph_no_sample - train_acc_subgraph_sample,
                        'val_diff': val_acc_fullgraph_no_sample - val_acc_subgraph_sample,
                        'test_diff': test_acc_fullgraph_no_sample - test_acc_subgraph_sample,

                        'train_acc_fullgraph_no_sample': train_acc_fullgraph_no_sample,
                        'val_acc_fullgraph_no_sample': val_acc_fullgraph_no_sample,
                        'test_acc_fullgraph_no_sample': test_acc_fullgraph_no_sample,

                        'train_acc_subgraph_no_sample': train_acc_subgraph_no_sample,
                        'val_acc_subgraph_no_sample': val_acc_subgraph_no_sample,
                        'test_acc_subgraph_no_sample': test_acc_subgraph_no_sample,
            })
            
    logger.debug("total time for {} epochs = {}".format(num_epochs, total_time))
    logger.debug("avg time per epoch = {}".format(total_time/num_epochs))
    return best_val_acc, model

if __name__ == "__main__":

    val_acc, model = train()
        