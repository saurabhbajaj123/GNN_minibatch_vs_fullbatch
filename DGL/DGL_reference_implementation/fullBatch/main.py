import os
os.environ['DGLBACKEND'] = 'pytorch'
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.data
import dgl.nn.pytorch as dglnn
from ogb.nodeproppred import DglNodePropPredDataset

root = "../dataset/"
dataset = DglNodePropPredDataset('ogbn-arxiv', root=root)
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"


graph, node_labels = dataset[0]
# Add reverse edges since ogbn-arxiv is unidirectional.
graph = dgl.add_reverse_edges(graph)
print(f"graph data = {graph.ndata}")

graph.ndata['label'] = node_labels[:, 0]

print(f"graph data keys = {graph.ndata.keys()}")

print(graph)
print(node_labels)

node_features = graph.ndata['feat']
num_features = node_features.shape[1]
num_classes = (node_labels.max() + 1).item()
num_layers = 6
num_hidden = 128
activation = F.relu
dropout = 0.5
print('Number of classes:', num_classes)

idx_split = dataset.get_idx_split()
train_nids = idx_split['train']
valid_nids = idx_split['valid']
test_nids = idx_split['test']

# print("len(train_nids) = {}, len(valid_nids) = {}, len(test_nids) = {} ".format(len(train_nids), len(valid_nids), len(test_nids)))
# graph.ndata['train_mask'] = train_nids
# graph.ndata['val_mask'] = valid_nids
# graph.ndata['test_mask'] = test_nids



class SAGE(nn.Module):
    def __init__(
        self, in_feats, n_hidden, n_classes, n_layers, activation, dropout
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, "mean"))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, "mean"))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, "mean"))
        # self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, g, x):
        h = x
        for l, conv in enumerate(self.layers):
            h = conv(g, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                # h = self.dropout(h)
        return h

    def inference(self, g, x, batch_size, device):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.
        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        h = x
        for l, conv in enumerate(self.layers):
            h = conv(g, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)

        return h

model = SAGE(num_features, num_hidden, num_classes, num_layers, activation, dropout).to(device)
opt = torch.optim.Adam(model.parameters())

def train(g, model, train_mask, val_mask, test_mask):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_val_acc = 0
    best_test_acc = 0

    features = g.ndata["feat"]
    labels = g.ndata["label"]
    for e in range(1000):
        # Forward
        logits = model(g, features)

        # Compute prediction
        pred = logits.argmax(1)

        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        # Compute accuracy on training/validation/test
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 5 == 0:
            print(
                "In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})".format(
                    e, loss, val_acc, best_val_acc, test_acc, best_test_acc
                )
            )


# model = GCN(g.ndata["feat"].shape[1], 16, dataset.num_classes)
train(graph, model, train_nids, valid_nids, test_nids)