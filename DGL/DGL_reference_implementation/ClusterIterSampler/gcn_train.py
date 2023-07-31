import time

import dgl
import dgl.nn as dglnn
from dgl.nn import SAGEConv


import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
from ogb.nodeproppred import DglNodePropPredDataset
from models import *
import wandb
wandb.login()
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train():
    wandb.init(
        project="GCN-cluster-{}".format(args.dataset),
        config={
            "num_epochs": 500,
            "lr": 1e-3,
            "dropout": random.uniform(0.0, 0.5),
            "n_hidden": 256,
            "n_layers": 3,
            "agg": "mean",
            "batch_size": 256,
            "num_parts": 5000,
            })
    
    config = wandb.config

    n_layers = config.n_layers
    n_hidden = config.n_hidden
    num_epochs = config.num_epochs
    dropout = config.dropout
    batch_size = config.batch_size
    lr = config.lr
    agg = config.agg
    num_partitions = config.num_parts
    activation = F.relu

    root="../dataset/"
    dataset = dgl.data.AsNodePredDataset(DglNodePropPredDataset("ogbn-arxiv", root=root))
    graph = dataset[
        0
    ]  # already prepares ndata['label'/'train_mask'/'val_mask'/'test_mask']

    device = "cuda" if torch.cuda.is_available() else "cpu"

    node_labels = graph.ndata['label']
    node_features = graph.ndata['feat']
    in_feats = node_features.shape[1]
    n_classes = (node_labels.max() + 1).item()

    model = GCN(in_feats, n_hidden, n_classes, n_layers, dropout).to(device)

    # model = SAGE(graph.ndata["feat"].shape[1], n_hidden, dataset.num_classes).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(opt, mode='max', factor=0.95, patience=10, min_lr=1e-5)

    sampler = dgl.dataloading.ClusterGCNSampler(
        graph,
        num_partitions,
        prefetch_ndata=["feat", "label", "train_mask", "val_mask", "test_mask"],
    )
    # DataLoader for generic dataloading with a graph, a set of indices (any indices, like
    # partition IDs here), and a graph sampler.
    dataloader = dgl.dataloading.DataLoader(
        graph,
        torch.arange(num_partitions).to("cuda"),
        sampler,
        device="cuda",
        batch_size=100,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=True,
    )

    durations = []
    
    best_train_acc = 0
    best_eval_acc = 0
    best_test_acc = 0

    for epoch in range(num_epochs):
        t0 = time.time()
        model.train()
        for it, sg in enumerate(dataloader):
            x = sg.ndata["feat"]
            y = sg.ndata["label"]
            m = sg.ndata["train_mask"].bool()
            y_hat = model(sg, x)
            loss = F.cross_entropy(y_hat[m], y[m])
            opt.zero_grad()
            loss.backward()
            opt.step()
            # if it % 20 == 0:
            #     acc = MF.accuracy(
            #         y_hat[m],
            #         y[m],
            #         task="multiclass",
            #         num_classes=dataset.num_classes,
            #     )
            #     mem = torch.cuda.max_memory_allocated() / 1000000
            #     print("Loss", loss.item(), "Acc", acc.item(), "GPU Mem", mem, "MB")
        # tt = time.time()
        # print(tt - t0)
        # durations.append(tt - t0)
        scheduler.step(best_eval_acc)

        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                train_preds, val_preds, test_preds = [], [], []
                train_labels, val_labels, test_labels = [], [], []
                for it, sg in enumerate(dataloader):
                    x = sg.ndata["feat"]
                    y = sg.ndata["label"]
                    m_train = sg.ndata["train_mask"].bool()
                    m_val = sg.ndata["val_mask"].bool()
                    m_test = sg.ndata["test_mask"].bool()
                    y_hat = model(sg, x)
                    train_preds.append(y_hat[m_train])
                    train_labels.append(y[m_train])
                    val_preds.append(y_hat[m_val])
                    val_labels.append(y[m_val])
                    test_preds.append(y_hat[m_test])
                    test_labels.append(y[m_test])
                train_preds = torch.cat(train_preds, 0)
                train_labels = torch.cat(train_labels, 0)
                val_preds = torch.cat(val_preds, 0)
                val_labels = torch.cat(val_labels, 0)
                test_preds = torch.cat(test_preds, 0)
                test_labels = torch.cat(test_labels, 0)

                train_acc = MF.accuracy(
                    train_preds,
                    train_labels,
                    task="multiclass",
                    num_classes=dataset.num_classes,
                )
                val_acc = MF.accuracy(
                    val_preds,
                    val_labels,
                    task="multiclass",
                    num_classes=dataset.num_classes,
                )
                test_acc = MF.accuracy(
                    test_preds,
                    test_labels,
                    task="multiclass",
                    num_classes=dataset.num_classes,
                )
                # print("Validation acc:", val_acc.item(), "Test acc:", test_acc.item())
                
                
                if best_eval_acc < val_acc:
                    best_eval_acc = val_acc
                    best_test_acc = test_acc
                    best_train_acc = train_acc
                print('Epoch {}, Train Acc {:.4f} (Best {:.4f}), Val Acc {:.4f} (Best {:.4f}), Test Acc {:.4f} (Best {:.4f})'.format(epoch, train_acc, best_train_acc, val_acc, best_eval_acc, test_acc, best_test_acc))
                    
                wandb.log({'val_acc': val_acc,
                    'test_acc': test_acc,
                    'train_acc': train_acc,
                    'best_eval_acc': best_eval_acc,
                    'best_test_acc': best_test_acc,
                    'best_train_acc': best_train_acc,
                    'lr': opt.param_groups[0]['lr'],
                })
    print(np.mean(durations[4:]), np.std(durations[4:]))

if __name__ == "__main__":
    
    # train()
        
    sweep_configuration = {
        'method': 'random',
        'metric': {'goal': 'maximize', 'name': 'val_acc'},
        'parameters': 
        {
            'n_hidden': {'distribution': 'int_uniform', 'min': 256, 'max': 2048},
            'n_layers': {'distribution': 'int_uniform', 'min': 3, 'max': 10},
            # 'lr': {'distribution': 'uniform', 'max': 2e-3, 'min': 1e-4},
            # 'dropout': {'distribution': 'uniform', 'min': 0.5, 'max': 0.8},
            "agg": {'values': ["mean", "gcn", "pool"]},
            # 'num_epochs': {'values': [2000, 4000, 6000, 8000]},
            # 'batch_size': {'values': [128, 256, 512]},
            # 'num_parts': {'distribution': 'int_uniform', 'min': 1000, 'max': 10000},
        }
    }
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='GCN-cluster-{}'.format(args.dataset))

    wandb.agent(sweep_id, function=train, count=30)