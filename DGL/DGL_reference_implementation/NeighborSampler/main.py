import dgl
import torch
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset
import time 

root = "../dataset/"
dataset = DglNodePropPredDataset('ogbn-arxiv', root=root)
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
graph, node_labels = dataset[0]
# Add reverse edges since ogbn-arxiv is unidirectional.
graph = dgl.add_reverse_edges(graph)
graph.ndata['label'] = node_labels[:, 0]
print(graph)
print(node_labels)

node_features = graph.ndata['feat']
num_features = node_features.shape[1]
num_classes = (node_labels.max() + 1).item()
num_layers = 6
num_hidden = 128
print('Number of classes:', num_classes)

idx_split = dataset.get_idx_split()
train_nids = idx_split['train']
valid_nids = idx_split['valid']
test_nids = idx_split['test']

sampler = dgl.dataloading.NeighborSampler([4 for _ in range(num_layers)])
train_dataloader = dgl.dataloading.DataLoader(
    # The following arguments are specific to DGL's DataLoader.
    graph,              # The graph
    train_nids,         # The node IDs to iterate over in minibatches
    sampler,            # The neighbor sampler
    device=device,      # Put the sampled MFGs on CPU or GPU
    # The following arguments are inherited from PyTorch DataLoader.
    batch_size=1024,    # Batch size
    shuffle=True,       # Whether to shuffle the nodes for every epoch
    drop_last=False,    # Whether to drop the last incomplete batch
    num_workers=0       # Number of sampler processes
)

input_nodes, output_nodes, mfgs = example_minibatch = next(iter(train_dataloader))
print(example_minibatch)
print("To compute {} nodes' outputs, we need {} nodes' input features".format(len(output_nodes), len(input_nodes)))

mfg_0_src = mfgs[0].srcdata[dgl.NID]
mfg_0_dst = mfgs[0].dstdata[dgl.NID]
print(mfg_0_src)
print(mfg_0_dst)
print(torch.equal(mfg_0_src[:mfgs[0].num_dst_nodes()], mfg_0_dst))

import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv

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

model = Model(num_features, num_hidden, num_classes, num_layers).to(device)

opt = torch.optim.Adam(model.parameters())

valid_dataloader = dgl.dataloading.DataLoader(
    graph, valid_nids, sampler,
    batch_size=1024,
    shuffle=False,
    drop_last=False,
    num_workers=0,
    device=device
)

test_dataloader = dgl.dataloading.DataLoader(
    graph, test_nids, sampler,
    batch_size=1024,
    shuffle=False,
    drop_last=False,
    num_workers=0,
    device=device
)
import tqdm
import sklearn.metrics

best_accuracy = 0
best_model_path = 'model.pt'

num_epochs = 5
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
            accuracy = sklearn.metrics.accuracy_score(labels, predictions)
            if best_accuracy < accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), best_model_path)
            print('Epoch {}, Val Acc {}, Best Val Acc {}'.format(epoch, accuracy, best_accuracy))

            # Note that this tutorial do not train the whole model to the end.
            # break

print("total time for {} epochs = {}".format(num_epochs, total_time))

model.load_state_dict(torch.load(best_model_path))
model.eval()
predictions = []
labels = []
# with tqdm.tqdm(valid_dataloader) as tq, torch.no_grad():
with torch.no_grad():
    # for input_nodes, output_nodes, mfgs in tq:
    for input_nodes, output_nodes, mfgs in test_dataloader:
        inputs = mfgs[0].srcdata['feat']
        labels.append(mfgs[-1].dstdata['label'].cpu().numpy())
        predictions.append(model(mfgs, inputs).argmax(1).cpu().numpy())
    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)
    accuracy = sklearn.metrics.accuracy_score(labels, predictions)
    print('Test Acc {}'.format(accuracy))
