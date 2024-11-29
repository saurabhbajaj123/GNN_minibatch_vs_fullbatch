from torch import nn
import torch
import math

import dgl.function as fn


class GraphSAGELayer(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 bias=True,
                 use_pp=False):
        super(GraphSAGELayer, self).__init__()
        self.use_pp = use_pp
        if self.use_pp:
            self.linear = nn.Linear(2 * in_feats, out_feats, bias=bias)
        else:
            self.linear1 = nn.Linear(in_feats, out_feats, bias=bias)
            self.linear2 = nn.Linear(in_feats, out_feats, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        if self.use_pp:
            stdv = 1. / math.sqrt(self.linear.weight.size(1))
            self.linear.weight.data.uniform_(-stdv, stdv)
            if self.linear.bias is not None:
                self.linear.bias.data.uniform_(-stdv, stdv)
        else:
            stdv = 1. / math.sqrt(self.linear1.weight.size(1))
            self.linear1.weight.data.uniform_(-stdv, stdv)
            self.linear2.weight.data.uniform_(-stdv, stdv)
            if self.linear1.bias is not None:
                self.linear1.bias.data.uniform_(-stdv, stdv)
                self.linear2.bias.data.uniform_(-stdv, stdv)

    def forward(self, graph, feat, in_deg):
        flops = 0
        with graph.local_scope():
            if self.training:
                if self.use_pp:
                    feat = self.linear(feat)
                else:
                    degs = in_deg.unsqueeze(1)
                    num_dst = graph.num_nodes('_V')
                    graph.nodes['_U'].data['h'] = feat
                    graph['_E'].update_all(fn.copy_u('h', out='m'),
                                           fn.sum(msg='m', out='h'),
                                           etype='_E')
                    ah = graph.nodes['_V'].data['h'] / degs
                    feat = self.linear1(feat[0:num_dst]) + self.linear2(ah)
                    # print(f"num_dst = {num_dst}, ah.shape = {ah.shape}, feat.shape = {feat.shape}, len(degs) = {len(degs)}")
                    # print(f"num_dst = {num_dst}, sum(degs) = {sum(degs)}")
                    flops = calculate_graphsage_flops_full_graph_per_layer(ah.shape[1], feat.shape[1], degs, num_dst)
                    # print(f"flops = {flops}")
                    # flops = gcn_flops(ah.shape[1], feat.shape[1], degs, num_dst)
            else:
                assert in_deg is None
                degs = graph.in_degrees().unsqueeze(1)
                graph.ndata['h'] = feat
                graph.update_all(fn.copy_u('h', out='m'),
                                 fn.sum(msg='m', out='h'))
                ah = graph.ndata.pop('h') / degs
                if self.use_pp:
                    feat = self.linear(torch.cat((feat, ah), dim=1))
                else:
                    feat = self.linear1(feat) + self.linear2(ah)
        return feat, flops



def calculate_graphsage_flops_full_graph_per_layer(F_in, F_out, in_degs, num_dst):
    n1 = (num_dst + sum(in_degs))
    n2 = (2*num_dst + sum(in_degs))
    total_flops = 0
    total_flops += 2 * n1 * F_in * F_out  # Matrix multiplication
    total_flops += n2 * F_out  # Aggregation
    return total_flops / 1e12


def gat_flops(F_in, F_out, in_degs, num_dst, num_heads):
    num_edges = sum(in_degs)

    return num_heads * (num_edges)*(6*F_in*F_out + 6*F_out + 2) / 1e12

def gcn_flops(F_in, F_out, in_degs, num_dst):
    num_edges = sum(in_degs)
    return (2*F_in*num_edges + 2*F_in*F_out*num_dst + num_dst*F_in)/1e12