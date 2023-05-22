import dgl
import torch

import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
import torch.optim as optim
import sklearn.metrics
import numpy as np

dgl.distributed.initialize(ip_config='ip_config.txt')
torch.distributed.init_process_group(backend='gloo')

g = dgl.distributed.DistGraph(graph_name='part_config', part_config ='partition_dataset/part_config.json')

# g = dgl.distributed.DistGraph(graph_name='part_config')

train_nid = dgl.distributed.node_split(g.ndata['train_mask'])
val_nid = dgl.distributed.node_split(g.ndata['val_mask'])
test_nid = dgl.distributed.node_split(g.ndata['test_mask'])


class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))

    def forward(self, blocks, x):
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            x = layer(block, x)
            if l != self.n_layers - 1:
                x = F.relu(x)
        return x

num_hidden = 256
num_labels = len(th.unique(g.ndata['labels'][0:g.number_of_nodes()]))
num_layers = 2
lr = 0.001
model = SAGE(g.ndata['feat'].shape[1], num_hidden, num_labels, num_layers)
loss_fcn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


model = th.nn.parallel.DistributedDataParallel(model)



train_dataloader = dgl.dataloading.DistNodeDataLoader(
                                g, train_nid, sampler, batch_size=1024,
                                shuffle=True, drop_last=False)
val_dataloader = dgl.dataloading.DistNodeDataLoader(
                                g, val_nid, sampler, batch_size=1024,
                                shuffle=False, drop_last=False)
test_dataloader = dgl.dataloading.DistNodeDataLoader(
                                g, test_nid, sampler, batch_size=1024,
                                shuffle=False, drop_last=False)

for epoch in range(10):
    # Loop over the dataloader to sample mini-batches.
    losses = []
    with model.join():
        for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
            # Load the input features as well as output labels
            batch_inputs = g.ndata['feat'][input_nodes]
            batch_labels = g.ndata['labels'][seeds]

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.detach().cpu().numpy())
            optimizer.step()

    # validation
    predictions = []
    labels = []
    with th.no_grad(), model.join():
        for step, (input_nodes, seeds, blocks) in enumerate(valid_dataloader):
            inputs = g.ndata['feat'][input_nodes]
            labels.append(g.ndata['labels'][seeds].numpy())
            predictions.append(model(blocks, inputs).argmax(1).numpy())
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        accuracy = sklearn.metrics.accuracy_score(labels, predictions)
        print('Epoch {}: Validation Accuracy {}'.format(epoch, accuracy))
