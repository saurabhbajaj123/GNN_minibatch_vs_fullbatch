import argparse
import json
import logging
import os
import sys

import dgl
import torch
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset
import time 

import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv
import tqdm
import sklearn.metrics

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Model(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, num_layers, aggregator_type='mean'):
        super(Model, self).__init__()
        self.conv = []
        # self.conv1 = SAGEConv(in_feats, h_feats, aggregator_type='mean')
        self.conv.append(SAGEConv(in_feats, h_feats, aggregator_type=aggregator_type))
        for _ in range(num_layers - 2):
            self.conv.append(SAGEConv(h_feats, h_feats, aggregator_type=aggregator_type))
        self.conv.append(SAGEConv(h_feats, num_classes, aggregator_type=aggregator_type))

        # self.conv2 = SAGEConv(h_feats, num_classes, aggregator_type='mean')
        self.h_feats = h_feats
        self.conv = nn.ModuleList(self.conv)
        self.num_layers = num_layers

    def forward(self, mfgs, x):
        # Lines that are changed are marked with an arrow: "<---"

        # h_dst = x[:mfgs[0].num_dst_nodes()]  # <---
        # h = self.conv1(mfgs[0], (x, h_dst))  # <---
        # h = F.relu(h)
        # h_dst = h[:mfgs[1].num_dst_nodes()]  # <---
        # h = self.conv2(mfgs[1], (h, h_dst))  # <---
        h_dst = x[:mfgs[0].num_dst_nodes()]  # <---
        h = self.conv[0](mfgs[0], (x, h_dst))
        for i in range(1, self.num_layers - 1):
            h_dst = h[:mfgs[i].num_dst_nodes()]  # <---
            h = self.conv[i](mfgs[i], (h, h_dst))
            h = F.relu(h)
        h_dst = h[:mfgs[-1].num_dst_nodes()]  # <---
        h = self.conv[-1](mfgs[-1], (h, h_dst))
        return h


def _get_data_loader(batch_size=1024, sampler, device):
    logger.info("Get train data loader")
    root = "../dataset/"
    dataset = DglNodePropPredDataset('ogbn-arxiv', root=root)

    idx_split = dataset.get_idx_split()
    train_nids = idx_split['train']
    valid_nids = idx_split['valid']
    test_nids = idx_split['test']

    graph, node_labels = dataset[0]
    graph = dgl.add_reverse_edges(graph)
    graph.ndata['label'] = node_labels[:, 0]

    node_features = graph.ndata['feat']
    num_features = node_features.shape[1]
    num_classes = (node_labels.max() + 1).item()
    
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

    valid_dataloader = dgl.dataloading.DataLoader(
    graph, valid_nids, sampler,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=0,
    device=device
    )

    test_dataloader = dgl.dataloading.DataLoader(
    graph, test_nids, sampler,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=0,
    device=device
    )

    return (train_dataloader, valid_dataloader, test_dataloader, (num_features, num_classes))

def train(args):
    batch_size = args.batch_size
    num_layers = args.num_layers
    fanout = args.fanout
    num_hidden =args.num_hidden
    num_epochs = args.num_epochs

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sampler = dgl.dataloading.NeighborSampler([fanout for _ in range(num_layers)])

    train_dataloader, valid_dataloader, test_dataloader, (num_features, num_classes) = _get_data_loader(batch_size, sampler, device)

    input_nodes, output_nodes, mfgs = example_minibatch = next(iter(train_dataloader))

    model = Model(num_features, num_hidden, num_classes, num_layers).to(device)
    opt = torch.optim.Adam(model.parameters())

    best_accuracy = 0
    best_model_path = 'model.pt'

    num_epochs = 5
    total_time = 0

    for epoch in range(num_epochs):
        model.train()
        for step, (input_nodes, output_nodes, mfgs) in enumerate(train_dataloader):
            inputs = mfgs[0].srcdata['feat']
            labels = mfgs[-1].dstdata['label']
            tic = time.time()
            predictions = model(mfgs, inputs)
            loss = F.cross_entropy(predictions, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tac = time.time()
            total_time += (tac - tic)
            accuracy = sklearn.metrics.accuracy_score(labels.cpu().numpy(), predictions.argmax(1).detach().cpu().numpy())

        if epoch % 1 == 0:
            model.eval()

            predictions = []
            labels = []
            # with tqdm.tqdm(valid_dataloader) as tq, torch.no_grad():
            with torch.no_grad():
                # for input_nodes, output_nodes, mfgs in tq:
                for input_nodes, output_nodes, mfgs in valid_dataloader:
                
                    inputs = mfgs[0].srcdata['feat']
                    labels.append(mfgs[-1].dstdata['label'].cpu().numpy())
                    predictions.append(model(mfgs, inputs).argmax(1).cpu().numpy())
                predictions = np.concatenate(predictions)
                labels = np.concatenate(labels)
                accuracy = sklearn.metrics.accuracy_score(labels, predictions)
                if best_accuracy < accuracy:
                    best_accuracy = accuracy
                    torch.save(model.state_dict(), best_model_path)
                logger.debug('Epoch {}, Val Acc {}, Best Val Acc {}'.format(epoch, accuracy, best_accuracy))

    logger.debug("total time for {} epochs = {}".format(num_epochs, total_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
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

    parser.add_argument("--num_layers", type=int, default=16)
    parser.add_argument("--fanout", type=int, default=4)
    parser.add_argument("--num_hidden", type=int, default=128)
