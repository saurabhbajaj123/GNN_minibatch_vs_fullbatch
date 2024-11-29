import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_area_auto_adjustable
import numpy as np
import math

fig, ax = plt.subplots(1,2, figsize=(9, 4))
# # fig.suptitle('Time to accuracy')

n_gpus = [2,4,8,16,32,64]

pubmed = [9.629703242, 12.28300531, 13.54671473, 15.25235356, 16.29184891, 18.98771359]
arxiv = [8.629703242,9.872629646,11.4061043,13.15601867,15.29184891,16.92961279]
reddit = [4.05045893, 4.864113469, 6.526911343, 7.38952103, 9.639559122, 13.26370399]
products = [0.4583688739, 0.5773595213, 0.7305047996, 0.833059085, 0.8943019325, 1.027807212]
papers = [19.27579894, 22.35751868, 27.3285929, 36.48291403, 45.28544861, 52.75447528]
ax[0].plot(n_gpus, pubmed,  marker='+')
ax[0].plot(n_gpus, arxiv, marker='+')
ax[0].plot(n_gpus, reddit,  marker='+')
ax[0].plot(n_gpus, products, marker='+')
ax[0].plot(n_gpus, papers, marker='+')

from matplotlib.ticker import MultipleLocator

# fig.supxlabel('Num Partitions', size=20)
ax[0].set_xlabel('Num Partitions', fontsize=23)

fig.supylabel('FG/MB ratio', size=20)

ax[0].set_yscale('log')
# plt.legend(fontsize=10)
# plt.tight_layout()
# plt.show()


# n hidden vary

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

fg_dfs = []
n_gpus = 4
for dataset in ['pubmed', 'ogbn-arxiv', 'reddit', 'ogbn-products', 'ogbn-papers100m']:
    for n_hidden in [64, 128, 256, 512, 1024]:
        with open(f"data_transfer/{dataset}_{n_gpus}.json") as f:
            data = json.load(f)
            num_vertices_transferrred = sum(data.values())
            factor = ((config[dataset]['n_layers'] - 1)*n_hidden) + config[dataset]['in_feats']
            total_data = num_vertices_transferrred*(factor)
            df = pd.DataFrame({'dataset': [dataset], 'num_vert':num_vertices_transferrred, 'n_gpus':n_gpus, 'total_data':[total_data]},)
            fg_dfs.append(df)
fg = pd.concat(fg_dfs)

fg_dfs = []
for dataset in ['pubmed', 'ogbn-arxiv', 'reddit', 'ogbn-products', 'ogbn-papers100m']:
    for n_gpus in [4]:
        with open(f"simulate_minibatch_dataload/{dataset}/p{n_gpus}.json") as f:
            data = json.load(f)
            # num_vertices_transferrred = sum(data.values())
            num_vertices_transferrred = sum(data['results'][3]['results'][10]['num_loc_miss'])
            factor = config[dataset]['in_feats']
            total_data = num_vertices_transferrred*(factor)
            df = pd.DataFrame({'dataset': [dataset], 'num_vert':num_vertices_transferrred, 'n_gpus':n_gpus, 'total_data':[total_data]},)
            fg_dfs.append(df)

mb = pd.concat(fg_dfs)

n_hidden = [64, 128, 256, 512, 1024]

fg_epochs = {
    'pubmed': 100,
    'ogbn-arxiv': 280,
    'reddit': 110,
    'ogbn-products': 230,
    'ogbn-papers100m': 870,
}

mb_epochs = {
    'pubmed': [45,50,50,55,60,65],
    'ogbn-arxiv': [9,10,11,13,15,17],
    'reddit': [4,4,5,7,9,11],
    'ogbn-products': [25, 30, 35, 45, 55, 65],
    'ogbn-papers100m': [35,35,40,45,50,55],
}

dataset_names = {
    'pubmed': 'Pubmed',
    'ogbn-arxiv': 'Ogbn-arxiv',
    'reddit': 'Reddit',
    'ogbn-products': 'Ogbn-products',
    'ogbn-papers100m': 'Ogbn-papers100M',
}

# fig, ax = plt.subplots(1,1, figsize=(5, 3))
for dataset in ['pubmed', 'ogbn-arxiv', 'reddit', 'ogbn-products', 'ogbn-papers100m']:
    print(dataset)
    fgtransfer = fg[fg['dataset'] == dataset]['total_data'].tolist()
    print(fgtransfer)
    # *fg_epochs[dataset]
    mbtransfer = mb[mb['dataset'] == dataset]['total_data'].values[0]
    print(mbtransfer)
    plot_values = [(x*fg_epochs[dataset])/(mbtransfer*mb_epochs[dataset][i]) for i, x in enumerate(fgtransfer)]
    print(plot_values)
    ax[1].plot(n_hidden, plot_values, label=dataset_names[dataset], marker='+')

fig.supylabel('FG/MB ratio', size=23)

ax[1].set_yscale('log')
ax[1].set_xlabel('Hidden size', fontsize=23)

for i in range(2):
    ax[i].tick_params(axis='both', which='major', labelsize=15)

fig.legend(loc='upper center', fancybox=True, ncol=3,  fontsize=17, bbox_to_anchor=(0.5, 1.0))
plt.tight_layout()
plt.show()