import os
os.environ['DGLBACKEND'] = 'pytorch'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau

import random
import wandb
wandb.login()
import dgl
import dgl.data
import dgl.nn.pytorch as dglnn
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.nn import GATConv

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

def train():
    root = "../dataset/"
    dataset = DglNodePropPredDataset('ogbn-products', root=root)
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    # train(dataset, device)

    graph, node_labels = dataset[0]
    graph = graph.to(device)
    node_labels = node_labels.to(device)
    # # Add reverse edges since ogbn-products is unidirectional.
    graph = dgl.add_reverse_edges(graph)

    graph.ndata['label'] = node_labels[:, 0]

    node_features = graph.ndata['feat']
    in_feats = node_features.shape[1]
    n_classes = (node_labels.max() + 1).item()

    idx_split = dataset.get_idx_split()
    train_mask = idx_split['train']
    val_mask = idx_split['valid']
    test_mask = idx_split['test']

    
    wandb.init(
        project="full-batch-products-gat",
        config={
            "epochs": 500,
            "lr": 1e-3,
            "n_hidden": 512,
            "n_layers": 3,
            "num_heads": 2,
            })

    config = wandb.config
    print(config)
    num_heads = config.num_heads
    n_hidden = config.n_hidden
    n_layers = config.n_layers
    model = GAT(in_feats, num_heads, n_hidden, n_classes, n_layers).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    best_val_acc = 0
    best_test_acc = 0
    best_train_acc = 0

    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, eta_min=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.95, patience=30, min_lr=1e-4)
    

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

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(best_val_acc)

        if e % 5 == 0:
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
            print(
                "In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})".format(
                    e, loss, val_acc, best_val_acc, test_acc, best_test_acc
                )
            )

            wandb.log({'val_acc': val_acc,
                        'test_acc': test_acc,
                        'train_acc': train_acc,
                        'best_train_acc': best_train_acc,
                        'best_val_acc': best_val_acc,
                        'best_test_acc': best_test_acc,
                        # 'lr': scheduler.get_last_lr()[0],
                        'lr': optimizer.param_groups[0]['lr'],
            })

train()

# sweep_configuration = {
#     'method': 'bayes',
#     'metric': {'goal': 'maximize', 'name': 'val_acc'},
#     'parameters': 
#     {
#         # 'lr': {'distribution': 'log_uniform_values', 'min': 1e-3, 'max': 1e-1},
#         # 'num_heads': {'distribution': 'int_uniform', 'min': 1, 'max': 10},
#         'n_layers': {'distribution': 'int_uniform', 'min': 3, 'max': 10},
#         # 'n_hidden': {'distribution': 'int_uniform', 'min': 64, 'max': 1024},
#      }
# }
# sweep_id = wandb.sweep(sweep=sweep_configuration, project='full-batch-products-gat')

# wandb.agent(sweep_id, function=train, count=10)


