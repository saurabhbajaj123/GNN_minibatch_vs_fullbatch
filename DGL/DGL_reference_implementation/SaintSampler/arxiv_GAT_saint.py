import argparse
import json
import logging
import os
import sys
import pickle

import dgl
from dgl.nn import GATConv

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
# from dgl.nn import SAGEConv
import tqdm
import sklearn.metrics

from models import GAT
from utils import _get_data_loader

import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def train():
    
    wandb.init(
        project="mini-batch-arxiv-GAT-saint",
        config={
            "num_epochs": 1000,
            "lr": 2*1e-3,
            "dropout": random.uniform(0.5, 0.80),
            "n_hidden": 256,
            "n_layers": 3,
            "num_heads":2,
            "agg": "gcn",
            "batch_size": 2**10,
            "fanout": 4,
            "budget": 1000,
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
    num_heads = config.num_heads
    
    root="../dataset/"
    dataset = DglNodePropPredDataset('ogbn-arxiv', root=root)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # sampler = dgl.dataloading.NeighborSampler([fanout for _ in range(n_layers)])
    sampler = dgl.dataloading.SAINTSampler(mode='node', budget=budget)

    data = _get_data_loader(sampler, device, dataset, batch_size)

    train_dataloader, valid_dataloader, test_dataloader, (in_feats, n_classes) = data

    # input_nodes, output_nodes, mfgs = example_minibatch = next(iter(train_dataloader))

    activation = F.relu

    model = GAT(in_feats, num_heads, n_hidden, n_classes, n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, eta_min=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.95, patience=10, min_lr=1e-5)

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
        # print("epoch = {}".format(epoch))
        model.train()
        tic = time.time()
        
        

        for step, subg in enumerate(train_dataloader):
            # print(step)
            tic_start = time.time()
            inputs = subg.ndata['feat']
            labels = subg.ndata['label']
            tic_step = time.time()
            # print("tic_step= {}".format(tic_step))
            predictions = model(subg, inputs)
            loss = F.cross_entropy(predictions, labels)
            optimizer.zero_grad()
            tic_forward = time.time()
            # print("tic_forward = {}".format(tic_forward))
            loss.backward()
            optimizer.step()
            tic_backward = time.time()
            # print("tic_backward = {}".format(tic_backward))

            time_load += tic_step - tic_start
            time_forward += tic_forward - tic_step
            time_backward += tic_backward - tic_forward

            # accuracy = sklearn.metrics.accuracy_score(labels.cpu().numpy(), predictions.argmax(1).detach().cpu().numpy())
            # if step % 100 == 0:
            #     logger.debug(
            #             "Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f}".format(
            #                 epoch, step, loss.item(), accuracy.item()
            #             )
            #         )
                # print(
                #         "Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f}".format(
                #             epoch, step, loss.item(), accuracy.item()
                #         )
                #     )
        # print("1 batch over")
        scheduler.step(best_eval_acc)
        toc = time.time()
        total_time += toc - tic
        # logger.debug(
        #     "Epoch Time(s): {:.4f} Load {:.4f} Forward {:.4f} Backward {:.4f}".format(
        #         toc - tic, time_load, time_forward, time_backward
        #     )
        # )        
        # print(
        #     "Epoch Time(s): {:.4f} Load {:.4f} Forward {:.4f} Backward {:.4f}".format(
        #         toc - tic, time_load, time_forward, time_backward
        #     )
        # )

        if epoch % 5 == 0:
            model.eval()
            # print("evalua")
            train_predictions = []
            train_labels = []
            val_predictions = []
            val_labels = []
            test_predictions = []
            test_labels = []
            with torch.no_grad():
                # print('start evauation')
                for subg in train_dataloader:
                    inputs = subg.ndata['feat']
                    train_labels.append(subg.ndata['label'].cpu().numpy())
                    train_predictions.append(model(subg, inputs).argmax(1).cpu().numpy())
                train_predictions = np.concatenate(train_predictions)
                train_labels = np.concatenate(train_labels)
                train_acc = sklearn.metrics.accuracy_score(train_labels, train_predictions)
                
                for subg in valid_dataloader:
                    # print(subg)
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

                if best_eval_acc < eval_acc:
                    best_eval_acc = eval_acc
                    best_model = model
                    best_test_acc = test_acc
                    best_train_acc = train_acc
                print('Epoch {}, Train Acc {:.4f} (Best {:.4f}), Val Acc {:.4f} (Best {:.4f}), Test Acc {:.4f} (Best {:.4f})'.format(epoch, train_acc, best_train_acc, eval_acc, best_eval_acc, test_acc, best_test_acc))
            
            wandb.log({'val_acc': eval_acc,
                        'test_acc': test_acc,
                        'train_acc': train_acc,
                        'best_eval_acc': best_eval_acc,
                        'best_test_acc': best_test_acc,
                        'best_train_acc': best_train_acc,
                        # 'lr': scheduler.get_last_lr()[0],
                        'lr': optimizer.param_groups[0]['lr'],

            })
            
    print("total time for {} epochs = {}".format(num_epochs, total_time))
    print("avg time per epoch = {}".format(total_time/num_epochs))
    return best_eval_acc, model

if __name__ == "__main__":
    
    # args = parse_args_fn()

    # eval_acc, model = train()
        
    
    sweep_configuration = {
        'method': 'bayes',
        'metric': {'goal': 'maximize', 'name': 'val_acc'},
        'parameters': 
        {
            'n_hidden': {'distribution': 'int_uniform', 'min': 256, 'max': 2048},
            'n_layers': {'distribution': 'int_uniform', 'min': 3, 'max': 10},
            'num_heads': {'distribution': 'int_uniform', 'min': 3, 'max': 10},
            # 'dropout': {'distribution': 'uniform', 'min': 0.5, 'max': 0.8},
            # "agg": {'values': ["mean", "gcn", "pool"]},
            # 'num_epochs': {'values': [2000, 4000, 6000, 8000]},
            # 'batch_size': {'values': [128, 256, 512]},
            'budget': {'distribution': 'int_uniform', 'min': 100, 'max': 10000},
        }
    }
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='mini-batch-arxiv-GAT-saint')

    wandb.agent(sweep_id, function=train, count=30)