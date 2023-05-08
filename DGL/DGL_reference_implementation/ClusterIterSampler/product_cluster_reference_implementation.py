import time

import dgl
import dgl.nn as dglnn
from dgl.nn import SAGEConv


import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
import wandb
wandb.login()
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau


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

class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, "mean"))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, "mean"))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, "mean"))
        self.dropout = nn.Dropout(0.5)

    def forward(self, sg, x):
        h = x
        for l, layer in enumerate(self.layers):
            h = layer(sg, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h
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
        project="mini-batch-cluster-products",
        config={
            "num_epochs": 2000,
            "lr": 5*1e-4,
            "dropout": random.uniform(0.0, 0.2),
            "n_hidden": 706,
            "n_layers": 7,
            "agg": "mean",
            "batch_size": 128,
            "num_parts": 1264,
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

    root=".././dataset/"
    dataset = DglNodePropPredDataset("ogbn-products", root=root)
    
    idx_split = dataset.get_idx_split()
    train_nids = idx_split['train']
    valid_nids = idx_split['valid']
    test_nids = idx_split['test']
    
    dataset = dgl.data.AsNodePredDataset(dataset)

    graph = dataset[
        0
    ]  # already prepares ndata['label'/'train_mask'/'val_mask'/'test_mask']
    
    evaluator = Evaluator(name='ogbn-products')

    graph = dgl.add_reverse_edges(graph)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    node_labels = graph.ndata['label']
    node_features = graph.ndata['feat']
    in_feats = node_features.shape[1]
    n_classes = (node_labels.max() + 1).item()

    model = Model(in_feats, n_hidden, n_classes, n_layers, dropout, activation, aggregator_type=agg).to(device)

    # model = SAGE(graph.ndata["feat"].shape[1], n_hidden, dataset.num_classes).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(opt, mode='max', cooldown=10, factor=0.95, patience=10, min_lr=1e-4)

    sampler = dgl.dataloading.ClusterGCNSampler(
        graph,
        num_partitions,
        prefetch_ndata=["feat", "label", "train_mask", "val_mask", "test_mask"],
        cache_path='train_cluster_gcn_{}_{}_{}.pkl'.format(n_layers, n_hidden, num_partitions),
    )
    # DataLoader for generic dataloading with a graph, a set of indices (any indices, like
    # partition IDs here), and a graph sampler.
    dataloader = dgl.dataloading.DataLoader(
        graph,
        torch.arange(num_partitions).to("cuda"),
        sampler,
        device=device,
        batch_size=batch_size,
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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
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
                
                device = "cpu"
                model = model.to(device)
                pred = model(graph.to(device), graph.ndata['feat'].to(device))
                train_acc_fullgraph_no_sample = evaluate(evaluator, pred[train_nids].argmax(1), graph.ndata['label'][train_nids].to(device))
                val_acc_fullgraph_no_sample = evaluate(evaluator, pred[valid_nids].argmax(1), graph.ndata['label'][valid_nids].to(device))
                test_acc_fullgraph_no_sample = evaluate(evaluator, pred[test_nids].argmax(1), graph.ndata['label'][test_nids].to(device))

                wandb.log({'val_acc': val_acc,
                    'test_acc': test_acc,
                    'train_acc': train_acc,
                    'best_eval_acc': best_eval_acc,
                    'best_test_acc': best_test_acc,
                    'best_train_acc': best_train_acc,
                    'train_acc_fullgraph_no_sample': train_acc_fullgraph_no_sample,
                    'val_acc_fullgraph_no_sample': val_acc_fullgraph_no_sample,
                    'test_acc_fullgraph_no_sample': test_acc_fullgraph_no_sample,
                    'lr': opt.param_groups[0]['lr'],
                })
    # print(np.mean(durations[4:]), np.std(durations[4:]))

if __name__ == "__main__":
    
    train()
    
    # sweep_configuration = {
    #     'method': 'bayes',
    #     'metric': {'goal': 'maximize', 'name': 'val_acc'},
    #     'parameters': 
    #     {
    #         # 'lr': {'distribution': 'log_uniform_values', 'min': 5*1e-3, 'max': 1e-1},
    #         'n_hidden': {'distribution': 'int_uniform', 'min': 256, 'max': 1024},
    #         'n_layers': {'distribution': 'int_uniform', 'min': 3, 'max': 10},
    #         'num_parts': {'distribution': 'int_uniform', 'min': 1000, 'max': 10000}
    #     }
    # }
    # sweep_id = wandb.sweep(sweep=sweep_configuration, project='mini-batch-cluster-products')

    # wandb.agent(sweep_id, function=train, count=50)
