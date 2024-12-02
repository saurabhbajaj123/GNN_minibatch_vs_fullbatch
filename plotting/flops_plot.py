
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_area_auto_adjustable
import numpy as np
import math


pubmed =[45.58452557, 38.53527844, 35.49976609, 31.14178438, 26.81393406, 23.30539192, 19.86697709]
arxiv = [31.88405802, 19.14165512, 13.66480641, 9.56910681, 6.377089338, 5.609852128, 5.288270449]
reddit = [208.5434416, 208.5815888, 139.0291638, 138.6297816, 62.88176409, 45.76936631, 37.19746273]
products = [12.08013023, 9.664341435, 8.052250653, 6.902551435, 5.369952227, 4.390934873, 3.718876409]
papers = [1050.963196,788.4937046,630.56316,450.3631291,350.3146502,286.6110511,242.5216834]


n_gpus = [1,2,3,4,8,16,32]



fig, ax = plt.subplots(1,3, figsize=(12, 4))
# fig.suptitle('Time to accuracy')
fig.supxlabel('Num workers', size=25)
fig.supylabel('FG/MB ratio', size=25)
cols = ["GraphSAGE", "GAT", "GCN"]


for a, col in zip(ax, cols):
    a.set_title(col, size=20)
# for i in range(4):
    # for j in range(4):
    # ax[i, j].tick_params(axis='both', which='major', labelsize=6)
    # ax[i].tick_params(axis='both', labelsize=12)


# n_gpus = [1, 2, 3, 4]
ax[0].plot(n_gpus, pubmed, label='Pubmed', marker='+',)
ax[0].plot(n_gpus, arxiv, label='Ogbn-arxiv', marker='+')
ax[0].plot(n_gpus, reddit, label='Reddit', marker='+')
ax[0].plot(n_gpus, products, label='Ogbn-products', marker='+')
ax[0].plot(n_gpus, papers, label='Ogbn-papers100m', marker='+')

ax[0].set_ylim(0.99, 3000)
ax[0].set_yscale('log')

# ax[0].legend()

gcn_mb_flops = {
    'Pubmed': 0.0003932706022,
    'Ogbn-arxiv': 0.3894915879,
    'Reddit': 1.40548718,
    'Ogbn-products': 0.4878094792,
    'Ogbn-papers100M': 0.2598735094
}

gcn_mb_epochs = {
    'Pubmed': [40, 58, 61, 65, 80, 120, 150],
    'Ogbn-arxiv': [5, 8, 11, 15, 20, 25, 30],
    'Reddit': [8, 11, 15, 20, 30, 45, 60],
    'Ogbn-products': [20, 25, 30, 35, 45, 55, 70],
    'Ogbn-papers100M': [6, 7, 10, 15, 20, 25, 35]
}

gcn_fg_flops = {
    'Pubmed': 0.007901468314,
    'Ogbn-arxiv': 0.04005143791,
    'Reddit': 0.3633737862,
    'Ogbn-products': 0.6612501144,
    'Ogbn-papers100M': 1.084392,
    
}

gcn_fg_epochs = {
    'Pubmed': 120,
    'Ogbn-arxiv': 540,
    'Reddit': 960,
    'Ogbn-products': 400,
    'Ogbn-papers100M': 260
}

#### gat ####

gat_mb_flops = {
    'Pubmed': 0.016503097489476204,
    'Ogbn-arxiv': 153.65802001953125,
    'Reddit': 18.069686889648438,
    'Ogbn-products': 79.67467308044434,
    'Ogbn-papers100M': 6.779729843139648
}

gat_mb_epochs = {
    'Pubmed': [40, 43, 50, 60, 75, 80, 100],
    'Ogbn-arxiv': [10, 13, 15, 20, 25, 30, 40],
    'Reddit': [8, 11, 15, 15, 30, 45, 65],
    'Ogbn-products': [20, 25, 30, 35, 45, 55, 70],
    'Ogbn-papers100M': [7, 8, 11, 15, 25, 30, 40]
}
gat_fg_flops = {
    'Pubmed': 0.2533400356769562,
    'Ogbn-arxiv': 5.250518321990967*10,
    'Reddit': 45.44840393066406,
    'Ogbn-products':53.75321960449219,
    'Ogbn-papers100M': 76.449615
}

gat_fg_epochs = {
    'Pubmed': 80,
    'Ogbn-arxiv': 340,
    'Reddit': 960,
    'Ogbn-products':400,
    'Ogbn-papers100M': 260
}


# pubmed_gat = [(gat_fg_flops['Pubmed']*gat_fg_epochs['Pubmed'])/(gat_mb_flops['Pubmed']*epoch) for i, epoch in enumerate(gat_mb_epochs['Pubmed'])]

for dataset in ["Pubmed", "Ogbn-arxiv", "Reddit", "Ogbn-products", "Ogbn-papers100M"]:
    ax[1].plot(n_gpus, [(gat_fg_flops[dataset]*gat_fg_epochs[dataset])/(gat_mb_flops[dataset]*epoch) for i, epoch in enumerate(gat_mb_epochs[dataset])], marker='+')

ax[1].set_yscale('log')

ax[1].set_ylim(0.99, 3000)



for dataset in ["Pubmed", "Ogbn-arxiv", "Reddit", "Ogbn-products", "Ogbn-papers100M"]:
    ax[2].plot(n_gpus, [(gcn_fg_flops[dataset]*gcn_fg_epochs[dataset])/(gcn_mb_flops[dataset]*epoch) for i, epoch in enumerate(gcn_mb_epochs[dataset])], marker='+')

ax[2].set_yscale('log')

ax[2].set_ylim(0.99, 3000)

for i in range(3):
    ax[i].tick_params(axis='both', which='major', labelsize=15)

fig.legend(loc='upper center', fancybox=True, ncol=3,  fontsize=18, bbox_to_anchor=(0.5, 1.0))

plt.tight_layout()
plt.show()
