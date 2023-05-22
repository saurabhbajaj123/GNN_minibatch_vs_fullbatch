import os
os.environ['DGLBACKEND'] = 'pytorch'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import random
import wandb
wandb.login()
import dgl
import dgl.data
import dgl.nn.pytorch as dglnn
from ogb.nodeproppred import DglNodePropPredDataset
from models import *
from conf import *


def get_model_and_config(name):
    name = name.lower()
    if name == "gat":
        return GAT, GAT_CONFIG
    elif name == "graphsage":
        return SAGE, GRAPHSAGE_CONFIG
    
def train():
    root = "../dataset/"
    dataset = DglNodePropPredDataset('ogbn-products', root=root)
    torch.cuda.set_device(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device = {}".format(device))
    # device = "cpu"
    # train(dataset, device)

    graph, node_labels = dataset[0]
    graph = graph.to(device)
    node_labels = node_labels.to(device)
    # Add reverse edges since ogbn-products is unidirectional.
    graph = dgl.add_reverse_edges(graph)
    print(f"graph data = {graph.ndata}")

    graph.ndata['label'] = node_labels[:, 0]

    print(f"graph data keys = {graph.ndata.keys()}")

    print(graph)
    print(node_labels)

    node_features = graph.ndata['feat']
    in_feats = node_features.shape[1]
    n_classes = (node_labels.max() + 1).item()
    print('Number of classes:', n_classes)

    idx_split = dataset.get_idx_split()
    train_mask = idx_split['train']
    val_mask = idx_split['valid']
    test_mask = idx_split['test']

    
    wandb.init(
        project="full-batch",
        config={
            "model": "Graphsage",
            "epochs": 1000,
            "lr": 2*1e-3,
            "dropout": random.uniform(0.5, 0.8),
            "n_hidden": 256,
            "n_layers": 6,
            "num_heads": 2,
            "agg": "gcn",
            "activation": F.relu,
            })

    config = wandb.config
    print(config)
    GNN, extra_config = get_model_and_config(config.model)
    if config.model.lower() == 'gat':
        extra_config['extra_args'] = [config.num_heads]
    elif config.model.lower() == "graphsage":
        extra_config['extra_args'] = [F.relu, config.dropout, config.agg]

    model = GNN(in_feats, n_classes, config.n_hidden, config.n_layers, *extra_config['extra_args']).to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    best_val_acc = 0
    best_test_acc = 0
    best_train_acc = 0
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, eta_min=1e-4)
    features = graph.ndata["feat"].to(device)
    labels = graph.ndata["label"].to(device)
    for e in range(config.epochs):
        # Forward
        logits = model(graph, features)

        # Compute prediction
        pred = logits.argmax(1)

        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        # Compute accuracy on training/validation/test
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_train_acc = train_acc

        score = val_acc
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if e % 5 == 0:
            print(
                "In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})".format(
                    e, loss, val_acc, best_val_acc, test_acc, best_test_acc
                )
            )

            wandb.log({'val_acc': val_acc,
                        'test_acc': test_acc,
                        'train_acc': train_acc,
                        'best_val_acc': best_val_acc,
                        'best_test_acc': best_test_acc,
                        'best_train_acc': best_train_acc,
                        'lr': scheduler.get_last_lr()[0],
            })

if __name__ == "__main__":
    train()

    # sweep_configuration = {
    #     'method': 'grid',
    #     'metric': {'goal': 'maximize', 'name': 'val_acc'},
    #     'parameters': 
    #     {
    #         # 'lr': {'distribution': 'log_uniform_values', 'min': 1e-3, 'max': 1e-1},
    #         # 'num_hidden': {'distribution': 'int_uniform', 'min': 64, 'max': 1024},
    #         # 'num_layers': {'distribution': 'int_uniform', 'min': 3, 'max': 10},
    #         # 'dropout': {'distribution': 'uniform', 'min': 0.1, 'max': 0.8},
    #         'num_hidden': {'values': [512, 1024]},
    #         # "agg": {'values': ["mean", "gcn", "pool"]},
    #         # 'epochs': {'values': [2000, 4000, 6000, 8000, 10000]},

    #      }
    # }
    # sweep_id = wandb.sweep(sweep=sweep_configuration, project='full-batch')

    # wandb.agent(sweep_id, function=train, count=15)


