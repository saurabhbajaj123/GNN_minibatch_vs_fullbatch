import argparse

import dgl
import dgl.nn as dglnn

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import AddSelfLoop
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset
from models import *
from utils import load_data
from parser import create_parser
import time
import wandb
wandb.login()
import warnings
warnings.filterwarnings("ignore")

def evaluate(g, features, labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def train(g, features, labels, masks, model, args):
    # define train/val samples, loss function and optimizer
    train_mask = masks[0]
    val_mask = masks[1]
    test_mask = masks[2]
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    # training loop
    train_time = 0

    best_val_acc = 0
    best_test_acc = 0
    best_train_acc = 0
    for epoch in range(args.n_epochs):
        t0 = time.time()
        model.train()
        logits = model(g, features)
        # pred = torch.max(logits, dim=1)[1]
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t1 = time.time()
        acc = evaluate(g, features, labels, val_mask, model)
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                epoch, loss.item(), acc
            )
        )
        train_time += t1 - t0

        # if epoch % args.log_every == 0:
        #     with torch.no_grad():
        #         logits = model(g, features)
        #         pred = torch.max(logits, dim=1)[1]
        #         # train_acc = torch.sum(pred[train_mask] == labels[train_mask]).item() * 1.0 / len(labels)
        #         # val_acc = torch.sum(pred[val_mask] == labels[val_mask]).item() * 1.0 / len(labels)
        #         # test_acc = torch.sum(pred[test_mask] == labels[test_mask]).item() * 1.0 / len(labels)
        #         train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        #         val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        #         test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        #     print(
        #         "Epoch {:05d} | Loss {:.4f} | Train Acc {:.4f} | Val Acc {:.4f} | Test Acc {:.4f}".format(
        #             epoch, loss.item(), train_acc, val_acc, test_acc
        #         )
        #     )

        #     if best_val_acc < val_acc:
        #         best_val_acc = val_acc
        #         best_test_acc = test_acc
        #         best_train_acc = train_acc

        #     wandb.log({'val_acc': val_acc,
        #             'test_acc': test_acc,
        #             'train_acc': train_acc,
        #             'best_val_acc': best_val_acc,
        #             'best_test_acc': best_test_acc,
        #             'best_train_acc': best_train_acc,
        #             'train_time': train_time,
        #             'lr': optimizer.param_groups[0]['lr'],
        #     })

def main():
    args = create_parser()

    wandb.init(
        project="GCN-fullbatch-{}".format(args.dataset),
        config={
            "n_hidden": args.n_hidden,
            "n_layers": args.n_layers,
            "dropout": args.dropout,
            "lr": args.lr,
            }
    )
    config = wandb.config
    args.n_hidden = config.n_hidden
    args.n_layers = config.n_layers
    args.dropout = config.dropout
    args.lr = config.lr

    
    data = load_data(args.dataset)
    # transform = (
    #     AddSelfLoop()
    # )  # by default, it will first remove self-loops to prevent duplication
    # data = PubmedGraphDataset(transform=transform)
    g = data[0]
    if args.dataset == "ogbn-arxiv":
        g.edata.clear()
        g = dgl.to_bidirected(g, copy_ndata=True)
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
    else:
        g.edata.clear()
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = g.int().to(device)
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    masks = g.ndata["train_mask"], g.ndata["val_mask"], g.ndata["test_mask"]

    # create GCN model
    in_size = features.shape[1]
    out_size = data.num_classes
    model = GCN(in_size, args.n_hidden, out_size, args.n_layers).to(device)

    # model training
    print("Training...")
    train(g, features, labels, masks, model, args)

    # test the model
    print("Testing...")
    acc = evaluate(g, features, labels, masks[2], model)
    print("Test accuracy {:.4f}".format(acc))


if __name__ == "__main__":


    
    args = create_parser()

    main()

    # sweep_configuration = {
    #     # 'name': f"Multiple runs best parameters {args.n_gpus}",
    #     'name': f"agg vary2",
    #     # 'name': "checking if 5 layers is the best",
    #     'method': 'grid',
    #     'metric': {'goal': 'maximize', 'name': 'val_acc'},
    #     'parameters': 
    #     {
    #         # 'n_hidden': {'values': [128, 256, 512, 728, 1024]},
    #         # 'n_layers': {'values': [2, 3, 5, 7, 9]},
    #         # 'n_hidden': {'distribution': 'int_uniform', 'min': 64, 'max': 256},
    #         # 'n_layers': {'distribution': 'int_uniform', 'min': 3, 'max': 5},
    #         # 'dropout': {'distribution': 'uniform', 'min': 0.3, 'max': 0.8},
    #         # 'lr': {'distribution': 'uniform', 'min': 1e-3, 'max': 1e-2},
    #         "agg": {'values': ["mean", "gcn", "pool"]},
    #         # 'batch_size': {'values': [256, 512, 1024, 2048, 4096]},
    #         # 'n_gpus': {'values': [4,3,2,1]},
    #         # 'dummy': {'values': [1, 2, 3, 4, 5]},
    #         # 'fanout': {'distribution': 'int_uniform', 'min': 3, 'max': 10},
    #         # "fanout": {'values': [4, 7, 10, 15, 20]}
    #         # 'dummy': {'distribution': 'uniform', 'min': 3, 'max': 10},
    #     }
    # }
    # sweep_id = wandb.sweep(sweep=sweep_configuration,
    #                        project="MultiGPU-{}-{}-{}".format(args.dataset, args.model, args.sampling))

    # wandb.agent(sweep_id, function=main, count=5)

