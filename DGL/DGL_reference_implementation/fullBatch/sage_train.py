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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau

def evaluate(g, features, labels, masks, model):
    model.eval()
    with torch.no_grad():
        train_mask = masks[0]
        val_mask = masks[1]
        test_mask = masks[2]
        logits = model(g, features)

        val_logits = logits[val_mask]
        val_labels = labels[val_mask]
        train_logits = logits[train_mask]
        train_labels = labels[train_mask]
        test_logits = logits[test_mask]
        test_labels = labels[test_mask]


        _, val_indices = torch.max(val_logits, dim=1)
        val_correct = torch.sum(val_indices == val_labels)
        val_acc = val_correct.item() * 1.0 / len(val_labels)

        _, train_indices = torch.max(train_logits, dim=1)
        train_correct = torch.sum(train_indices == train_labels)
        train_acc = train_correct.item() * 1.0 / len(train_labels)

        _, test_indices = torch.max(test_logits, dim=1)
        test_correct = torch.sum(test_indices == test_labels)
        test_acc = test_correct.item() * 1.0 / len(test_labels)

        return train_acc, val_acc, test_acc


def train(g, features, labels, masks, model, args):

    if args.seed:
        torch.manual_seed(args.seed)
    # define train/val samples, loss function and optimizer
    train_mask = masks[0]
    val_mask = masks[1]
    test_mask = masks[2]
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.95, cooldown=10, patience=30, min_lr=1e-5)

    # training loop
    train_time = 0

    best_val_acc = 0
    best_test_acc = 0
    best_train_acc = 0
    no_improvement_count = 0

    for epoch in range(args.n_epochs):
        t0 = time.time()
        model.train()
        logits = model(g, features)
        # pred = torch.max(logits, dim=1)[1]
        pred = logits.argmax(1)

        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t1 = time.time()
        # acc = evaluate(g, features, labels, val_mask, model)
        # print(
        #     "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
        #         epoch, loss.item(), acc
        #     )
        # )
        train_time += t1 - t0
        # scheduler.step(best_val_acc)

        if epoch % args.log_every == 0:
            # train_acc, val_acc, test_acc = evaluate(g, features, labels, masks, model)

            train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
            val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
            test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

            t2 = time.time()
            print(
                "Epoch {:05d} | Loss {:.4f} | Train Acc {:.4f} | Val Acc {:.4f} | Test Acc {:.4f}".format(
                    epoch, loss.item(), train_acc, val_acc, test_acc
                )
            )
            print("Train time {:.4f}, Eval time {:.4f}".format(t1-t0, t2-t1))

            if best_val_acc < val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                best_train_acc = train_acc

                no_improvement_count = 0
            else:
                no_improvement_count += args.log_every

            wandb.log({'val_acc': val_acc,
                    'test_acc': test_acc,
                    'train_acc': train_acc,
                    'best_val_acc': best_val_acc,
                    'best_test_acc': best_test_acc,
                    'best_train_acc': best_train_acc,
                    'train_time': train_time,
                    'lr': optimizer.param_groups[0]['lr'],
            })
        if epoch > 50 and no_improvement_count >= args.patience:
            print(f'Early stopping after {epoch + 1} epochs.')
            break
def main():
    args = create_parser()

    wandb.init(
        project="SAGE-fullbatch-{}".format(args.dataset),
        config={
            "n_hidden": args.n_hidden,
            "n_layers": args.n_layers,
            "dropout": args.dropout,
            "lr": args.lr,
            "agg": args.agg,
            }
    )
    config = wandb.config
    args.n_hidden = config.n_hidden
    args.n_layers = config.n_layers
    args.dropout = config.dropout
    args.lr = config.lr
    args.agg = config.agg

    
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
        # g.edata.clear()
        g = dgl.remove_self_loop(g)
        # g = dgl.add_self_loop(g)
        # g = dgl.add_reverse_edges(g)

    device = torch.device("cuda:{}".format(args.device_id) if torch.cuda.is_available() else "cpu")
    # g = g.int().to(device)
    print(f"device = {device}")
    g = g.to(device)
    features = g.ndata["feat"].to(device)
    labels = g.ndata["label"].to(device)
    masks = g.ndata["train_mask"], g.ndata["val_mask"], g.ndata["test_mask"]

    # create GCN model
    in_size = features.shape[1]
    out_size = data.num_classes
    model = SAGE(in_size, args.n_hidden, out_size, args.n_layers, F.relu, args.dropout, args.agg).to(device)

    # model training
    print("Training...")
    train(g, features, labels, masks, model, args)

    wandb.log({
        "seed": args.seed,
    })

    # # test the model
    # print("Testing...")
    # _, _, acc = evaluate(g, features, labels, masks, model)
    # print("Test accuracy {:.4f}".format(acc))


if __name__ == "__main__":


    
    args = create_parser()

    # main()

    sweep_configuration = {
        # 'name': f"Multiple runs best parameters {args.n_gpus}",
        'name': f"n_layers vary",
        # 'name': "checking if 5 layers is the best",
        'method': 'grid',
        'metric': {'goal': 'maximize', 'name': 'val_acc'},
        'parameters':
        {
            'n_layers': {'values': [2,3,4,5,6]},
            'n_hidden': {'values': [256, 512, 1024]},
            # 'n_hidden': {'distribution': 'int_uniform', 'min': 32, 'max': 128},
            # 'n_layers': {'distribution': 'int_uniform', 'min': 3, 'max': 10},
            # 'dropout': {'distribution': 'uniform', 'min': 0.3, 'max': 0.8},
            # 'lr': {'distribution': 'uniform', 'min': 1e-4, 'max': 1e-2},
            # 'lr': {'values': [args.lr, args.lr*2]}
            # "agg": {'values': ["mean", "gcn", "pool"]},
            # 'batch_size': {'values': [256, 512, 1024, 2048, 4096]},
            # 'n_gpus': {'values': [4,3,2,1]},
            # 'dummy': {'values': [1, 2, 3, 4, 5]},
            # 'fanout': {'distribution': 'int_uniform', 'min': 3, 'max': 10},
            # "fanout": {'values': [4, 7, 10, 15, 20]}
            # 'dummy': {'distribution': 'uniform', 'min': 3, 'max': 10},
        }
    }
    sweep_id = wandb.sweep(sweep=sweep_configuration,
                           project="SAGE-fullbatch-{}".format(args.dataset))

    wandb.agent(sweep_id, function=main, count=1000)

