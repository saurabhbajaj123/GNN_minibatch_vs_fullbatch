import torch
import torch.nn.functional as F


GAT_CONFIG = {
    "extra_args": [2],
    "lr": 0.005,
}

GRAPHSAGE_CONFIG = {
    "extra_args": [F.relu, 0.5, "gcn"],
    "lr": 1e-3,
}