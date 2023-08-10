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

from parser import create_parser
from utils import load_data
from models import *

def _get_data_loader(sampler, device, graph, train_nids, batch_size=1024):
    
    logger.info("Get train data loader")
    train_dataloader = dgl.dataloading.DataLoader(
    # The following arguments are specific to DGL's DataLoader.
    graph,              # The graph
    torch.arange(len(train_nids)/batch_size),         # The node IDs to iterate over in minibatches
    sampler,            # The neighbor sampler
    device=device,      # Put the sampled MFGs on CPU or GPU
    # The following arguments are inherited from PyTorch DataLoader.
    batch_size=batch_size,    # Batch size
    shuffle=True,       # Whether to shuffle the nodes for every epoch
    drop_last=False,    # Whether to drop the last incomplete batch
    num_workers=4,      # Number of sampler processes
    # use_uva=True,
    )    
    return train_dataloader

@torch.no_grad()
def evaluate(evaluator, predictions, labels):
    # print(labels.size(), predictions.size())
    acc = evaluator.eval({
        'y_true': torch.reshape(labels, (-1, 1)),
        'y_pred': torch.reshape(predictions, (-1, 1)),
    })['acc']
    # eacc = sklearn.metrics.accuracy_score(labels, predictions)
    return acc

def main():
    args = create_parser()
    print(args)
    wandb.init(
        project="{}-SingleGPU-Saint-{}".format(args.model, args.dataset),
        config={
            "n_epochs": args.n_epochs,
            "lr": args.lr,
            "dropout": args.dropout,
            "n_hidden": args.n_hidden,
            "n_layers": args.n_layers,
            "agg": args.agg,
            "batch_size": args.batch_size,
            "budget_node_edge": args.budget_node_edge,
            "budget_rw_1": args.budget_rw_1,
            "budget_rw_0": args.budget_rw_0
            })


    config = wandb.config
    args.n_layers = config.n_layers
    args.n_hidden = config.n_hidden
    args.n_epochs = config.n_epochs
    args.dropout = config.dropout
    args.batch_size = config.batch_size
    args.lr = config.lr
    args.agg = config.agg
    args.budget_node_edge = config.budget_node_edge
    args.budget_rw_0 = config.budget_rw_0 
    args.budget_rw_1 = config.budget_rw_1 
    activation = F.relu
    
        
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
    if "sage" in args.model.lower():
        model = SAGE(in_feats, args.n_hidden, n_classes, args.n_layers, args.dropout, activation, aggregator_type=args.agg).to(device)
    elif "gcn" in args.model.lower():
        model = GCN(in_feats, args.n_hidden, n_classes, args.n_layers, args.dropout)
    elif "gat" in args.model.lower():
        model = GAT(in_feats, args.n_hidden, n_classes, args.n_layers, args.num_heads)
    train(graph, dataset, node_features, device, model, args)


def train(graph, dataset, node_features, device, model, args):

    # creating the sampler

    # sampler = dgl.dataloading.NeighborSampler([fanout for _ in range(n_layers)])
    # sampler = dgl.dataloading.SAINTSampler(
    #     mode='edge', 
    #     budget=args.budget_node_edge, 
    #     # prefetch_ndata=["feat", "label", "train_mask", "val_mask", "test_mask"]
    #     )
    if args.mode_saint == 'walk':
        budget = (args.budget_rw_0,args.budget_rw_1) 
    else:
        budget = args.budget_node_edge 
    print(args.mode_saint, budget)
    sampler = dgl.dataloading.SAINTSampler(
        mode=args.mode_saint,
        budget=budget,
        # prefetch_ndata=["feat", "label", "train_mask", "val_mask", "test_mask"],
    )


    # getting train, test, val splits
    train_nids = np.where(graph.ndata['train_mask'])[0]
    valid_nids = np.where(graph.ndata['val_mask'])[0]
    test_nids = np.where(graph.ndata['test_mask'])[0]

    train_dataloader = _get_data_loader(sampler, device, graph.subgraph(train_nids), train_nids, args.batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # scheduler1 = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, eta_min=1e-3)
    # scheduler2 = ReduceLROnPlateau(optimizer, mode='max', cooldown=20, factor=0.99, patience=30, min_lr=1e-5)

    evaluator = Evaluator(name='ogbn-arxiv')
    best_train_acc = 0
    best_val_acc = 0
    best_test_acc = 0

    best_model = None
    train_time = 0
    
    for epoch in range(args.n_epochs):
        t0 = time.time()
        device = "cuda:{}".format(args.device_id) if torch.cuda.is_available() else "cpu"
        model = model.to(device)
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
        t1 = time.time()
        train_time += t1 - t0

        # scheduler2.step(best_val_acc)
        # scheduler1.step()

        # Evaluating
        if (epoch + 1) % args.log_every == 0:
            model.eval()
            train_predictions = []
            train_labels = []
            val_predictions = []
            val_labels = []
            test_predictions = []
            test_labels = []
            with torch.no_grad():
                # Inference on sampled subgraph
                
                # for subg in train_dataloader:
                #     inputs = subg.ndata['feat']
                #     train_labels.append(subg.ndata['label'])
                #     train_predictions.append(model(subg, inputs).argmax(1))
                # train_predictions = torch.cat(train_predictions)
                # train_labels = torch.cat(train_labels)
                # train_acc = sklearn.metrics.accuracy_score(train_labels.cpu().numpy(), train_predictions.cpu().numpy())
            
                device = "cpu"
                model = model.to(device)
            
                # Inference on entire graph
                pred = model(graph, graph.ndata['feat'])

                train_acc_fullgraph_no_sample = evaluate(evaluator, pred[train_nids].argmax(1), graph.ndata['label'][train_nids])
                
                # # Inference on subgraph
                # pred_train = model(graph.subgraph(train_nids).to(device), graph.ndata['feat'][train_nids].to(device))
                # train_acc_subgraph_no_sample = evaluate(evaluator, pred_train.argmax(1), graph.ndata['label'][train_nids])
                
                # # Inference on sampled subgraph
                # for subg in valid_dataloader:
                #     inputs = subg.ndata['feat']
                #     val_labels.append(subg.ndata['label'])
                #     val_predictions.append(model(subg, inputs).argmax(1))
                # val_predictions = torch.cat(val_predictions)
                # val_labels = torch.cat(val_labels)
                # val_acc_subgraph_sample = sklearn.metrics.accuracy_score(val_labels.cpu().numpy(), val_predictions.cpu().numpy())
                
                val_acc_fullgraph_no_sample = evaluate(evaluator, pred[valid_nids].argmax(1), graph.ndata['label'][valid_nids])

                # # Inference on subgraph
                # pred_valid = model(graph.subgraph(valid_nids).to(device), graph.ndata['feat'][valid_nids].to(device))
                # val_acc_subgraph_no_sample = evaluate(evaluator, pred_valid.argmax(1), graph.ndata['label'][valid_nids])

                # # Inference on sampled subgraph
                # for subg in test_dataloader:
                #     inputs = subg.ndata['feat']
                #     test_labels.append(subg.ndata['label'])
                #     test_predictions.append(model(subg, inputs).argmax(1))
                # test_predictions = torch.cat(test_predictions)
                # test_labels = torch.cat(test_labels)
                # test_acc_subgraph_sample = sklearn.metrics.accuracy_score(test_labels.cpu().numpy(), test_predictions.cpu().numpy())
                
                test_acc_fullgraph_no_sample = evaluate(evaluator, pred[test_nids].argmax(1), graph.ndata['label'][test_nids])

                # # Inference on subgraph
                # pred_test = model(graph.subgraph(test_nids).to(device), graph.ndata['feat'][test_nids].to(device))
                # test_acc_subgraph_no_sample = evaluate(evaluator, pred_test.argmax(1), graph.ndata['label'][test_nids])
                

                t2 = time.time()

                eval_time = t2 -t1
                # Storing the best accuracies
                if best_val_acc < val_acc_fullgraph_no_sample:
                    best_val_acc = val_acc_fullgraph_no_sample
                    best_model = model
                    best_test_acc = test_acc_fullgraph_no_sample
                    best_train_acc = train_acc_fullgraph_no_sample
                print('Epoch {}, Train Acc {:.4f} (Best {:.4f}), Val Acc {:.4f} (Best {:.4f}), Test Acc {:.4f} (Best {:.4f})'.format(epoch, train_acc_fullgraph_no_sample, best_train_acc, val_acc_fullgraph_no_sample, best_val_acc, test_acc_fullgraph_no_sample, best_test_acc))
                print(f"Train time = {t1-t0}, Eval time = {t2-t1}")

            wandb.log({'val_acc': val_acc_fullgraph_no_sample,
                        'test_acc': test_acc_fullgraph_no_sample,
                        'train_acc': train_acc_fullgraph_no_sample,
                        # 'train_acc_fullgraph_no_sample': train_acc_fullgraph_no_sample,
                        'best_val_acc': best_val_acc,
                        'best_test_acc': best_test_acc,
                        'best_train_acc': best_train_acc,
                        # 'lr': scheduler.get_last_lr()[0],
                        'lr': optimizer.param_groups[0]['lr'],
                        'train_time': train_time,
            })
            
    # logger.debug("total time for {} epochs = {}".format(n_epochs, total_time))
    # logger.debug("avg time per epoch = {}".format(total_time/n_epochs))
    return best_val_acc, model

if __name__ == "__main__":

    # val_acc, model = train()
    # main()
    args = create_parser()
    sweep_configuration = {
        "name": "HPO",
        'method': 'bayes',
        'metric': {'goal': 'maximize', 'name': 'val_acc'},
        'parameters': 
        {
            # 'lr': {'distribution': 'log_uniform_values', 'min': 5*1e-4, 'max': 1e-1},
            'n_hidden': {'distribution': 'int_uniform', 'min': 64, 'max': 1024},
            'n_layers': {'distribution': 'int_uniform', 'min': 3, 'max': 10},
            # 'n_layers': {'values':[6, 7, 8]},
            'dropout': {'distribution': 'uniform', 'min': 0.2, 'max': 0.8},
            # "agg": {'values': ["mean", "gcn", "pool"]},
            # 'n_epochs': {'values': [2000, 4000, 6000, 8000]},
            # 'batch_size': {'distribution': 'int_uniform', 'min': 5, 'max': 10},
            # 'batch_size': {'values':[7, 6, 5]},
            # 'budget': {'distribution': 'int_uniform', 'min': 256, 'max': 1024},
            'num_heads': {'distribution': 'int_uniform', 'min': 1, 'max': 10},
        }
    }
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="{}-SingleGPU-Saint-{}".format(args.model, args.dataset),)

    wandb.agent(sweep_id, function=main, count=20)
