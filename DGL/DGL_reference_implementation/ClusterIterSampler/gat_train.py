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
import time
from utils import load_data
from parser import create_parser
import warnings
warnings.filterwarnings("ignore")
from torch.optim.lr_scheduler import ReduceLROnPlateau
def train(graph, dataset, node_features, node_labels, model, device, args):

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(opt, mode='max', factor=0.99, patience=20, min_lr=1e-4)

    sampler = dgl.dataloading.ClusterGCNSampler(
        graph,
        args.num_partitions,
        cache_path=f'cluster_gat_{args.dataset}_{args.n_hidden}_{args.n_layers}_{args.num_partitions}_{args.num_heads}.pkl',
        prefetch_ndata=["feat", "label", "train_mask", "val_mask", "test_mask"],
    )
    # DataLoader for generic dataloading with a graph, a set of indices (any indices, like
    # partition IDs here), and a graph sampler.
    dataloader = dgl.dataloading.DataLoader(
        graph,
        torch.arange(args.num_partitions).to(device),
        sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=True,
    )

    durations = []
    
    best_train_acc = 0
    best_eval_acc = 0
    best_test_acc = 0
    train_time = 0
    for epoch in range(args.n_epochs):
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
        t1 = time.time()
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
        train_time += t1 - t0
        # scheduler.step(best_eval_acc)

        if epoch % args.log_every == 0:
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
                
                t2 = time.time()
                if best_eval_acc < val_acc:
                    best_eval_acc = val_acc
                    best_test_acc = test_acc
                    best_train_acc = train_acc
                print('Epoch {}, Train Acc {:.4f} (Best {:.4f}), Val Acc {:.4f} (Best {:.4f}), Test Acc {:.4f} (Best {:.4f})'.format(epoch, train_acc, best_train_acc, val_acc, best_eval_acc, test_acc, best_test_acc))
                print(f"Train time = {t1-t0}, Eval time = {t2 - t1}")
                wandb.log({'val_acc': val_acc,
                    'test_acc': test_acc,
                    'train_acc': train_acc,
                    'best_eval_acc': best_eval_acc,
                    'best_test_acc': best_test_acc,
                    'best_train_acc': best_train_acc,
                    'train_time': train_time,
                    'lr': opt.param_groups[0]['lr'],
                })
    print(np.mean(durations[4:]), np.std(durations[4:]))

def main():
    args = create_parser()
    wandb.init(
        project="GAT-SingleGPU-cluster-{}".format(args.dataset),
        config={
            "n_epochs": args.n_epochs,
            "lr": args.lr,
            "dropout": args.dropout,
            "n_hidden": args.n_hidden,
            "n_layers": args.n_layers,
            "batch_size": args.batch_size,
            "agg": args.agg,
            "num_partitions": args.num_partitions,
            "num_heads": args.num_heads, 
            })
    
    config = wandb.config

    args.n_layers = config.n_layers
    args.n_hidden = config.n_hidden
    args.n_epochs = config.n_epochs
    args.dropout = config.dropout
    args.batch_size = config.batch_size
    args.lr = config.lr
    args.agg = config.agg
    args.num_partitions = config.num_partitions
    args.num_heads = config.num_heads

    # root="../dataset/"
    # dataset = dgl.data.AsNodePredDataset(DglNodePropPredDataset(args.dataset, root=root))
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

    node_labels = graph.ndata['label']
    node_features = graph.ndata['feat']
    in_feats = node_features.shape[1]
    num_classes = dataset.num_classes # (node_labels.max() + 1).item()

    model = GAT(in_feats, args.num_heads, args.n_hidden, num_classes, args.n_layers).to(device)

    # model = SAGE(in_feats, args.n_hidden, num_classes, args.n_layers, F.relu, args.dropout, args.agg).to(device)

    # model = SAGE(graph.ndata["feat"].shape[1], n_hidden, dataset.num_classes).cuda()
    
    train(graph, dataset, node_features, node_labels, model, device, args)
    
if __name__ == "__main__":
    
    main()
    args = create_parser()
    # sweep_configuration = {
    #     'name': "num_partitions",
    #     'method': 'random',
    #     'metric': {'goal': 'maximize', 'name': 'val_acc'},
    #     'parameters': 
    #     {
    #         # 'n_hidden': {'distribution': 'int_uniform', 'min': 256, 'max': 2048},
    #         # 'n_hidden': {'values': [256, 512, 728, 1024]},
    #         # 'n_layers': {'values': [2, 4, 6, 8, 10]},
    #         # 'n_layers': {'distribution': 'int_uniform', 'min': 3, 'max': 10},
    #         # 'lr': {'distribution': 'uniform', 'max': 5e-3, 'min': 5e-4},
    #         # 'dropout': {'distribution': 'uniform', 'min': 0.2, 'max': 0.8},
    #         # "agg": {'values': ["mean", "gcn", "pool"]},
    #         # 'n_epochs': {'values': [2000, 4000, 6000, 8000]},
    #         # 'batch_size': {'values': [128, 256, 512]},
    #         # 'num_partitions': {'distribution': 'int_uniform', 'min': 2000, 'max': 10000},
    #         # 'num_partitions': {'values': [4000, 6000, 8000]},
    #     }
    # }
    # sweep_id = wandb.sweep(sweep=sweep_configuration, project='SAGE-SingleGPU-cluster-{}'.format(args.dataset))

    # wandb.agent(sweep_id, function=main, count=10)