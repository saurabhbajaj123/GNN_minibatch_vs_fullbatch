import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_area_auto_adjustable
import numpy as np

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
for dataset in ['pubmed', 'ogbn-arxiv', 'reddit', 'ogbn-products', 'ogbn-papers100m']:
    for n_gpus in [4]:
        with open(r"G:\\My Drive\\Courses\\Sem_4_Spring_23\\Research\\report2\\data_transfer\\simulate_minibatch_dataload\\{}\p{}.json".format(dataset, n_gpus)) as f:
            data = json.load(f)
            # num_vertices_transferrred = sum(data.values())
            num_vertices_transferrred = sum(data['results'][3]['results'][10]['num_loc_miss'])
            factor = config[dataset]['in_feats']
            total_data = num_vertices_transferrred*(factor)
            df = pd.DataFrame({'dataset': [dataset], 'num_vert':num_vertices_transferrred, 'n_gpus':n_gpus, 'total_data':[total_data]},)
            dfs.append(df)
pd.concat(dfs).to_csv("data_transfer_hidden_result_MB.csv")
