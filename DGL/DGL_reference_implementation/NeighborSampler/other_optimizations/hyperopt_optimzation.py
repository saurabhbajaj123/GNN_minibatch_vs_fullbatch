from hyperopt import fmin, tpe, space_eval, Trials
from hyperopt import hp
from hyperopt.pyll.base import scope

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


def objective(args):

    root = "../dataset/"
    dataset = DglNodePropPredDataset('ogbn-arxiv', root=root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    graph, node_labels = dataset[0]
    # Add reverse edges since ogbn-arxiv is unidirectional.
    graph = dgl.add_reverse_edges(graph)
    graph.ndata['label'] = node_labels[:, 0]
    # print(graph)
    # print(node_labels)

    node_features = graph.ndata['feat']
    num_features = node_features.shape[1]
    num_classes = (node_labels.max() + 1).item()
    num_layers = args["num_layers"]
    num_hidden = args["num_hidden"]
    fanout = 4 # args["fanout"] # trial.suggest_int("fanout", 2, 8)
    sampler = dgl.dataloading.NeighborSampler([fanout for _ in range(num_layers)])
    print('Number of classes:', num_classes)

    idx_split = dataset.get_idx_split()
    train_nids = idx_split['train']
    valid_nids = idx_split['valid']
    test_nids = idx_split['test']

    train_dataloader = dgl.dataloading.DataLoader(
        # The following arguments are specific to DGL's DataLoader.
        graph,              # The graph
        train_nids,         # The node IDs to iterate over in minibatches
        sampler,            # The neighbor sampler
        device=device,      # Put the sampled MFGs on CPU or GPU
        # The following arguments are inherited from PyTorch DataLoader.
        batch_size=64,    # Batch size
        shuffle=True,       # Whether to shuffle the nodes for every epoch
        drop_last=False,    # Whether to drop the last incomplete batch
        num_workers=0       # Number of sampler processes
    )
    valid_dataloader = dgl.dataloading.DataLoader(
        graph, valid_nids, sampler,
        batch_size=64,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        device=device
    )

    # test_dataloader = dgl.dataloading.DataLoader(
    #     graph, test_nids, sampler,
    #     batch_size=512,
    #     shuffle=False,
    #     drop_last=False,
    #     num_workers=0,
    #     device=device
    # )

    input_nodes, output_nodes, mfgs = example_minibatch = next(iter(train_dataloader))
    # print(example_minibatch)
    print("To compute {} nodes' outputs, we need {} nodes' input features".format(len(output_nodes), len(input_nodes)))

    mfg_0_src = mfgs[0].srcdata[dgl.NID]
    mfg_0_dst = mfgs[0].dstdata[dgl.NID]
    print(mfg_0_src)
    print(mfg_0_dst)
    print(torch.equal(mfg_0_src[:mfgs[0].num_dst_nodes()], mfg_0_dst))



    class Model(nn.Module):
        def __init__(self, in_feats, h_feats, num_classes, num_layers):
            super(Model, self).__init__()
            self.conv = []
            # self.conv1 = SAGEConv(in_feats, h_feats, aggregator_type='mean')
            self.conv.append(SAGEConv(in_feats, h_feats, aggregator_type='mean'))
            for _ in range(num_layers - 2):
                self.conv.append(SAGEConv(h_feats, h_feats, aggregator_type='mean'))
            self.conv.append(SAGEConv(h_feats, num_classes, aggregator_type='mean'))

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

    model = Model(num_features, 2**num_hidden, num_classes, num_layers).to(device)

    opt = torch.optim.Adam(model.parameters())



    best_accuracy = 0
    best_val_accuracy = 0
    best_model_path = 'model.pt'

    num_epochs = 2
    total_time = 0
    
    for epoch in range(num_epochs):
        model.train()

        # with tqdm.tqdm(train_dataloader) as tq:
        for step, (input_nodes, output_nodes, mfgs) in enumerate(train_dataloader):
            # feature copy from CPU to GPU takes place here
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

            # tq.set_postfix({'loss': '%.03f' % loss.item(), 'acc': '%.03f' % accuracy}, refresh=False)
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
                val_accuracy = sklearn.metrics.accuracy_score(labels, predictions)
                if best_val_accuracy < val_accuracy:
                    best_val_accuracy = val_accuracy
                    torch.save(model.state_dict(), best_model_path)
                print('Epoch {}, Val Acc {}, Best Val Acc {}'.format(epoch, val_accuracy, best_val_accuracy))

                # Note that this tutorial do not train the whole model to the end.
                # break

    print("total time for {} epochs = {}".format(num_epochs, total_time))

    return -1*best_val_accuracy

param_space = {
    "num_layers": scope.int(hp.quniform("num_layers", 8, 16, 4)),
    "num_hidden": scope.int(hp.quniform("num_hidden", 6, 12, 4)),
    # "fanout": scope.int(hp.quniform("fanout", 2, 3, 1)),
}

trials = Trials()


best = fmin(objective, space=param_space, algo=tpe.suggest, max_evals=15, trials=trials)
print(best)