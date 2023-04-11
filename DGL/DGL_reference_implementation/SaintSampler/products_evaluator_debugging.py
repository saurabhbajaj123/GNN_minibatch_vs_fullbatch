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

def parse_args_fn():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        metavar="N",
        help="input batch size for training (default: 64)",
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )

    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--fanout", type=int, default=4)
    parser.add_argument("--n_hidden", type=int, default=2**6)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=20)

    # parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    # parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    # parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    # parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    # # parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    # parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    args = parser.parse_args()

    return args

def load_dataset(path):
    """
    Load entire dataset
    """
    # find all files with pkl extenstion and load the first one
    files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith("pkl")]

    if len(files) == 0:
        raise ValueError("Invalid # of files in dir: {}".format(path))

    dataset = pickle.load(open(files[0], 'rb'))

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
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=0,
    device=device
    )

    logger.info("Get test data loader")
    test_dataloader = dgl.dataloading.DataLoader(
    graph, test_nids, sampler,
    batch_size=batch_size,
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

@torch.no_grad()
def evaluate2(logits, labels, mask):
    logits = logits[mask]
    labels = labels[mask]
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)

def evaluate3(indices, labels):
    # _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)

def train():
    
    wandb.init(
        project="debugging",
        config={
            "num_epochs": 10,
            "lr": 2*1e-3,
            "dropout": random.uniform(0.3, 0.6),
            "n_hidden": 1024,
            "n_layers": 4,
            "agg": "mean",
            "batch_size": 2**10,
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
    
    root="../dataset/"
    dataset = DglNodePropPredDataset('ogbn-products', root=root)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    
    idx_split = dataset.get_idx_split()
    train_nids = idx_split['train']
    valid_nids = idx_split['valid']
    test_nids = idx_split['test']

    graph, node_labels = dataset[0]
    graph = dgl.add_reverse_edges(graph)
    graph.ndata['label'] = node_labels[:, 0]

    node_features = graph.ndata['feat']
    in_feats = node_features.shape[1]
    n_classes = (node_labels.max() + 1).item()

    # sampler = dgl.dataloading.NeighborSampler([fanout for _ in range(n_layers)])
    sampler = dgl.dataloading.SAINTSampler(mode='node', budget=budget)

    data = _get_data_loader(sampler, device, graph, (train_nids, valid_nids, test_nids), batch_size)

    train_dataloader, valid_dataloader, test_dataloader = data

    # input_nodes, output_nodes, mfgs = example_minibatch = next(iter(train_dataloader))

    activation = F.relu

    model = Model(in_feats, n_hidden, n_classes, n_layers, dropout, activation, aggregator_type=agg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, eta_min=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.99, patience=20, min_lr=1e-5)

    evaluator = Evaluator(name='ogbn-products')
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

                pred = model(graph.subgraph(train_nids).to(device), graph.ndata['feat'][train_nids].to(device))
                for subg in train_dataloader:
                    inputs = subg.ndata['feat']
                    train_labels.append(subg.ndata['label'])
                    train_predictions.append(model(subg, inputs).argmax(1))
                train_predictions = torch.cat(train_predictions)
                train_labels = torch.cat(train_labels)
                train_acc = sklearn.metrics.accuracy_score(train_labels.cpu().numpy(), train_predictions.cpu().numpy())
                # train_acc_thru_evaltr = evaluate2(pred, graph.ndata['label'].to(device), train_nids)
                train_acc_thru_evaltr = evaluate3(pred.argmax(1), graph.ndata['label'][train_nids].to(device))
                
                pred = model(graph.subgraph(valid_nids).to(device), graph.ndata['feat'][valid_nids].to(device))
                for subg in valid_dataloader:
                    inputs = subg.ndata['feat']
                    val_labels.append(subg.ndata['label'])
                    val_predictions.append(model(subg, inputs).argmax(1))
                val_predictions = torch.cat(val_predictions)
                val_labels = torch.cat(val_labels)
                eval_acc = sklearn.metrics.accuracy_score(val_labels.cpu().numpy(), val_predictions.cpu().numpy())
                # val_acc_thru_evaltr = evaluate2(pred, graph.ndata['label'].to(device), valid_nids)
                val_acc_thru_evaltr = evaluate3(pred.argmax(1), graph.ndata['label'][valid_nids].to(device))

                pred = model(graph.subgraph(test_nids).to(device), graph.ndata['feat'][test_nids].to(device))
                for subg in test_dataloader:
                    inputs = subg.ndata['feat']
                    test_labels.append(subg.ndata['label'])
                    test_predictions.append(model(subg, inputs).argmax(1))
                test_predictions = torch.cat(test_predictions)
                test_labels = torch.cat(test_labels)
                test_acc = sklearn.metrics.accuracy_score(test_labels.cpu().numpy(), test_predictions.cpu().numpy())
                # test_acc_thru_evaltr = evaluate2(pred, graph.ndata['label'].to(device), test_nids)
                test_acc_thru_evaltr = evaluate3(pred.argmax(1), graph.ndata['label'][test_nids].to(device))

                if best_eval_acc < eval_acc:
                    best_eval_acc = eval_acc
                    best_model = model
                    best_test_acc = test_acc
                    best_train_acc = train_acc
                logger.debug('Epoch {}, Train Acc {:.4f} (Best {:.4f}), Val Acc {:.4f} (Best {:.4f}), Test Acc {:.4f} (Best {:.4f})'.format(epoch, train_acc, best_train_acc, eval_acc, best_eval_acc, test_acc, best_test_acc))
            
            wandb.log({'val_acc': eval_acc,
                        'test_acc': test_acc,
                        'train_acc': train_acc,
                        'best_eval_acc': best_eval_acc,
                        'best_test_acc': best_test_acc,
                        'best_train_acc': best_train_acc,
                        # 'lr': scheduler.get_last_lr()[0],
                        'lr': optimizer.param_groups[0]['lr'],
                        'train_diff': train_acc_thru_evaltr - train_acc,
                        'val_diff': val_acc_thru_evaltr - eval_acc,
                        'test_diff': test_acc_thru_evaltr - test_acc,
                        'evltr_train': train_acc_thru_evaltr,
                        'evltr_val': val_acc_thru_evaltr,
                        'evltr_test': test_acc_thru_evaltr,
            })
            
    logger.debug("total time for {} epochs = {}".format(num_epochs, total_time))
    logger.debug("avg time per epoch = {}".format(total_time/num_epochs))
    return best_eval_acc, model

if __name__ == "__main__":
    
    # args = parse_args_fn()

    eval_acc, model = train()
        
    
    # sweep_configuration = {
    #     'method': 'bayes',
    #     'metric': {'goal': 'maximize', 'name': 'val_acc'},
    #     'parameters': 
    #     {
    #         'n_hidden': {'distribution': 'int_uniform', 'min': 256, 'max': 2048},
    #         'n_layers': {'distribution': 'int_uniform', 'min': 3, 'max': 10},
    #         # 'dropout': {'distribution': 'uniform', 'min': 0.5, 'max': 0.8},
    #         # "agg": {'values': ["mean", "gcn", "pool"]},
    #         # 'num_epochs': {'values': [2000, 4000, 6000, 8000]},
    #         # 'batch_size': {'values': [128, 256, 512]},
    #         # 'budget': {'distribution': 'int_uniform', 'min': 100, 'max': 10000},
    #     }
    # }
    # sweep_id = wandb.sweep(sweep=sweep_configuration, project='debugging')

    # wandb.agent(sweep_id, function=train, count=15)

#tmux
# ctrl+b -> d
# attach -t
# tmux attach -t 0