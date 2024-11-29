import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_area_auto_adjustable
import numpy as np
# pubmed = [6.452156668, 6.256353591, 6.332669323]
# arxiv = [0.1233945153,  0.1824318829, 0.1910933683]
# reddit = [0.1239705044, 0.116133746, 0.1149538204]
# products = [0.04432491351, 0.04626034872, 0.04922381626]
config= {
    'pubmed' : {
        'n_layers': 3,
        'n_hidden': 256,
        'in_feats': 500
    },

    'ogbn-arxiv' : {
        'n_layers': 2,
        'n_hidden': 512,
        'in_feats': 128
    },

    'reddit' : {
        'n_layers': 4,
        'n_hidden': 1024,
        'in_feats': 602
    },

    'ogbn-products' : {
        'n_layers': 5,
        'n_hidden': 256,
        'in_feats': 100
    },

    'ogbn-papers100m' : {
        'n_layers': 2,
        'n_hidden': 128,
        'in_feats': 128
    }
}
pubmed = []
arxiv = []
reddit = []
products = []

import json

import pandas as pd
dfs = []
n_gpus = 4
for dataset in ['pubmed', 'ogbn-arxiv', 'reddit', 'ogbn-products', 'ogbn-papers100m']:
    for n_hidden in [64, 128, 256, 512, 1024]:
        with open(f"./data_transfer/{dataset}_{n_gpus}.json") as f:
            data = json.load(f)
            num_vertices_transferrred = sum(data.values())
            factor = ((config[dataset]['n_layers'] - 1)*n_hidden) + config[dataset]['in_feats']
            total_data = num_vertices_transferrred*(factor)
            df = pd.DataFrame({'dataset': [dataset], 'num_vert':num_vertices_transferrred, 'n_gpus':n_gpus, 'total_data':[total_data]},)
            dfs.append(df)
pd.concat(dfs).to_csv("data_transfer_hidden_size_result.csv")




