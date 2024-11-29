import os

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import numpy as np
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from dgl.nn import SAGEConv, GATConv
import torch.distributed as dist
import dgl.nn
import dgl.nn.pytorch as dglnn

from ogb.nodeproppred import DglNodePropPredDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
)
from dgl.multiprocessing import shared_tensor


class SaintSAGE(nn.Module):
    def __init__(
        self, in_feats, n_hidden, n_classes, n_layers, dropout, activation):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_feats, n_hidden, "mean"))
        for i in range(1, n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, "mean"))
        self.layers.append(SAGEConv(n_hidden, n_classes, "mean"))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, g, x):
        h = x
        for l, conv in enumerate(self.layers):
            h = conv(g, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, x):
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

class ClusterSAGE(nn.Module):
    def __init__(
        self, in_feats, n_hidden, n_classes, n_layers, dropout, activation):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_feats, n_hidden, "mean"))
        for i in range(1, n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, "mean"))
        self.layers.append(SAGEConv(n_hidden, n_classes, "mean"))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, g, x):
        h = x
        for l, conv in enumerate(self.layers):
            h = conv(g, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
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



class SAGE(nn.Module):
    def __init__(
        self, in_feats, n_hidden, n_classes, n_layers, dropout, activation, aggregator_type='mean',
    ):
        super().__init__()
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

    def forward(self, mfgs, x):
        h_dst = x[:mfgs[0].num_dst_nodes()]  # <---
        h = self.layers[0](mfgs[0], (x, h_dst))
        for i in range(1, self.n_layers - 1):
            h_dst = h[:mfgs[i].num_dst_nodes()]  # <---
            h = self.layers[i](mfgs[i], (h, h_dst))
            # h = F.relu(h)
            h = self.activation(h)
            h = self.dropout(h)
        h_dst = h[:mfgs[-1].num_dst_nodes()]  # <---
        h = self.layers[-1](mfgs[-1], (h, h_dst))
        return h


    def inference(self, g, x):
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

class GAT(nn.Module):
    def __init__(
        self, in_feats, num_heads, n_hidden, n_classes, n_layers
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.num_heads = num_heads

        self.layers = nn.ModuleList()
        self.layers.append(GATConv(in_feats, n_hidden, num_heads=num_heads))
        for _ in range(n_layers - 2):
            self.layers.append(GATConv(n_hidden*num_heads, n_hidden, num_heads=num_heads))
        self.layers.append(GATConv(n_hidden*num_heads, n_classes, num_heads=1))


    def forward(self, g, x):
        h = x
        for i in range(self.n_layers - 1):
            h = self.layers[i](g, h)
            h = h.flatten(1)
        h = self.layers[-1](g, h)
        h = h.mean(1)
        return h



class GraphSAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, dropout, activation, aggregator_type='mean'):
        super().__init__()

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

    def forward(self, blocks, x):
        total_edges = 0
        total_dst_nodes = 0
        total_flops = 0
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            # print(f'block = {block.num_dst_nodes()}')
            # print(f"block = {block.num_edges()}")
            # total_edges += block.num_edges()
            # total_dst_nodes += block.num_dst_nodes()
            F_in = h.shape[1]
            h = layer(block, h)
            F_out = h.shape[1]
            # print(f"F_in = {F_in}, F_out = {F_out}")
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        # print(f"total_edges in the model= {total_edges}")
            total_flops += calculate_graphsage_flops_full_graph_per_layer(F_in, F_out, block.num_edges(), block.num_dst_nodes())
        # print(f"total_flops = {total_flops}")
        return h, torch.tensor(total_flops)

    # def forward(self, mfgs, x):
    #     h_dst = x[:mfgs[0].num_dst_nodes()]  # <---
    #     h = self.layers[0](mfgs[0], (x, h_dst))
    #     for i in range(1, self.n_layers - 1):
    #         h_dst = h[:mfgs[i].num_dst_nodes()]  # <---
    #         h = self.layers[i](mfgs[i], (h, h_dst))
    #         # h = F.relu(h)
    #         h = self.activation(h)
    #         h = self.dropout(h)
    #     h_dst = h[:mfgs[-1].num_dst_nodes()]  # <---
    #     h = self.layers[-1](mfgs[-1], (h, h_dst))
    #     return h


    def inference(self, g, device, batch_size, use_uva):
        g.ndata["h"] = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["h"])
        for l, layer in enumerate(self.layers):
            dataloader = DataLoader(
                g,
                torch.arange(g.num_nodes(), device=device),
                sampler,
                device=device,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=0,
                use_ddp=True,
                use_uva=use_uva,
            )
            # in order to prevent running out of GPU memory, allocate a
            # shared output tensor 'y' in host memory
            y = shared_tensor(
                (
                    g.num_nodes(),
                    self.n_hidden
                    if l != len(self.layers) - 1
                    else self.n_classes,
                )
            )
            # for input_nodes, output_nodes, blocks in (
            #     tqdm.tqdm(dataloader) if dist.get_rank() == 0 else dataloader
            # ):
            for input_nodes, output_nodes, blocks in (dataloader):
                x = blocks[0].srcdata["h"]
                h = layer(blocks[0], x)  # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # non_blocking (with pinned memory) to accelerate data transfer
                y[output_nodes] = h.to(y.device, non_blocking=True)
            # make sure all GPUs are done writing to 'y'
            dist.barrier()
            g.ndata["h"] = y if use_uva else y.to(device)

        g.ndata.pop("h")
        return y

    # def inference(self, g, x):
    #     """
    #     Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
    #     g : the entire graph.
    #     x : the input of entire node set.
    #     The inference code is written in a fashion that it could handle any number of nodes and
    #     layers.
    #     """
    #     # During inference with sampling, multi-layer blocks are very inefficient because
    #     # lots of computations in the first few layers are repeated.
    #     # Therefore, we compute the representation of all nodes layer by layer.  The nodes
    #     # on each layer are of course splitted in batches.
    #     # TODO: can we standardize this?
    #     h = x
    #     for l, conv in enumerate(self.layers):
    #         h = conv(g, h)
    #         if l != len(self.layers) - 1:
    #             h = self.activation(h)

    #     return h


class GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, dropout=0.5):
        super().__init__()
        
        self.layers = nn.ModuleList()
        # two-layer GCN
        self.layers.append(dglnn.GraphConv(in_feats, n_hidden, activation=F.relu))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.GraphConv(n_hidden, n_hidden, activation=F.relu))

        self.layers.append(dglnn.GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)

        return h


class NSGCN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout=0.5):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        # two-layer GCN
        self.layers.append(dglnn.GraphConv(in_feats, n_hidden, activation=F.relu))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.GraphConv(n_hidden, n_hidden, activation=F.relu))

        self.layers.append(dglnn.GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
    
    def forward(self, blocks, x):
        h = x
        total_flops = 0
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            F_in = h.shape[1]
            h = layer(block, h)
            F_out = h.shape[1]

            total_flops += gcn_flops(F_in, F_out, block.num_edges(), block.num_dst_nodes())

            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h, total_flops

    # def forward(self, mfgs, x):
    #     h_dst = x[:mfgs[0].num_dst_nodes()]  # <---
    #     h = self.layers[0](mfgs[0], (x, h_dst))
    #     for i in range(1, self.n_layers - 1):
    #         h_dst = h[:mfgs[i].num_dst_nodes()]  # <---
    #         h = self.layers[i](mfgs[i], (h, h_dst))
    #         # h = F.relu(h)
    #         h = self.activation(h)
    #         h = self.dropout(h)
    #     h_dst = h[:mfgs[-1].num_dst_nodes()]  # <---
    #     h = self.layers[-1](mfgs[-1], (h, h_dst))
    #     return h


    def inference(self, g, device, batch_size, use_uva):
        g.ndata["h"] = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["h"])
        for l, layer in enumerate(self.layers):
            dataloader = DataLoader(
                g,
                torch.arange(g.num_nodes(), device=device),
                sampler,
                device=device,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=0,
                use_ddp=True,
                use_uva=use_uva,
            )
            # in order to prevent running out of GPU memory, allocate a
            # shared output tensor 'y' in host memory
            y = shared_tensor(
                (
                    g.num_nodes(),
                    self.n_hidden
                    if l != len(self.layers) - 1
                    else self.n_classes,
                )
            )
            # for input_nodes, output_nodes, blocks in (
            #     tqdm.tqdm(dataloader) if dist.get_rank() == 0 else dataloader
            # ):
            for input_nodes, output_nodes, blocks in (dataloader):
                x = blocks[0].srcdata["h"]
                h = layer(blocks[0], x)  # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # non_blocking (with pinned memory) to accelerate data transfer
                y[output_nodes] = h.to(y.device, non_blocking=True)
            # make sure all GPUs are done writing to 'y'
            dist.barrier()
            g.ndata["h"] = y if use_uva else y.to(device)

        g.ndata.pop("h")
        return y




class NSGAT(nn.Module):
    def __init__(
        self, in_feats, num_heads, n_hidden, n_classes, n_layers, dropout=0.5
    ):
        super(NSGAT, self).__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.num_heads = num_heads

        self.layers = nn.ModuleList()
        self.layers.append(GATConv(in_feats, n_hidden, num_heads=num_heads))
        for _ in range(n_layers - 2):
            self.layers.append(GATConv(n_hidden*num_heads, n_hidden, num_heads=num_heads))
        self.layers.append(GATConv(n_hidden*num_heads, n_classes, num_heads=1))

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, mfgs, x):
        # h_dst = x[:mfgs[0].num_dst_nodes()]  # <---
        # h = self.layers[0](mfgs[0], x)
        h = x
        total_flops = 0
        for i in range(self.n_layers - 1):
            # h_dst = h[:mfgs[i].num_dst_nodes()]  # <---
            # print(mfgs[i], h.shape)
            F_in = h.shape[-1]
            h = self.layers[i](mfgs[i], h)
            F_out = h.shape[-1]
            # print(self.layers[i].num_heads())
            # print(self.layers[i].num_heads)
            # h = F.relu(h)
            # h = self.activation(h)
            # h = self.dropout(h)
            h = h.flatten(1)
            total_flops += gat_flops(F_in, F_out, mfgs[i].num_edges(), mfgs[i].num_dst_nodes(), self.num_heads)

        F_in = h.shape[-1]
        # h_dst = h[:mfgs[-1].num_dst_nodes()]  # <---
        h = self.layers[-1](mfgs[-1], h)
        F_out = h.shape[-1]
        total_flops += gat_flops(F_in, F_out, mfgs[i].num_edges(), mfgs[i].num_dst_nodes(), self.num_heads)

        # print(h.shape)
        h = h.mean(1)
        # print(h.shape)
        return h, total_flops


    def inference(self, g, device, batch_size, use_uva):
        g.ndata["h"] = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["h"])
        for l, layer in enumerate(self.layers):
            dataloader = DataLoader(
                g,
                torch.arange(g.num_nodes(), device=device),
                sampler,
                device=device,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=0,
                use_ddp=True,
                use_uva=use_uva,
            )
            # in order to prevent running out of GPU memory, allocate a
            # shared output tensor 'y' in host memory
            y = shared_tensor(
                (
                    g.num_nodes(),
                    self.n_hidden*self.num_heads
                    if l != len(self.layers) - 1
                    else self.n_classes,
                )
            )
            # print(y.shape)
            # for input_nodes, output_nodes, blocks in (
            #     tqdm.tqdm(dataloader) if dist.get_rank() == 0 else dataloader
            # ):
            for input_nodes, output_nodes, blocks in (dataloader):
                x = blocks[0].srcdata["h"]
                h = layer(blocks[0], x)  # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = h.flatten(1)
                    # h = F.relu(h)
                    # h = self.dropout(h)
                # h = h.mean(1)
                # non_blocking (with pinned memory) to accelerate data transfer
                else:
                    h = h.mean(1)
                # print(f"h = {h.shape}")
                y[output_nodes] = h.to(y.device, non_blocking=True)
            # make sure all GPUs are done writing to 'y'
            dist.barrier()
            g.ndata["h"] = y if use_uva else y.to(device)

        g.ndata.pop("h")
        return y



def calculate_graphsage_flops_full_graph_per_layer(F_in, F_out, num_edges, num_dst):
    n1 = (num_dst + num_edges)
    n2 = (2*num_dst + num_edges)
    total_flops = 0
    total_flops += 2 * n1 * F_in * F_out  # Matrix multiplication
    total_flops += n2 * F_out  # Aggregation
    total_flops = num_dst*(4*F_in*F_out) + num_edges*2*F_in
    return total_flops / 1e12

def gat_flops(F_in, F_out, num_edges, num_dst, num_heads):
    return num_heads * (num_edges)*(6*F_in*F_out + 6*F_out + 2) / 1e12

def gcn_flops(F_in, F_out, num_edges, num_dst):
    return (2*F_in*num_edges + 2*F_in*F_out*num_dst + num_dst*F_in)/1e12