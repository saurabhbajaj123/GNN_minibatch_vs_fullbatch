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


# G:\My Drive\Courses\Sem_4_Spring_23\Research\report2\data_transfer\simulate_minibatch_dataload\pubmed\1.json
# G:\My Drive\Courses\Sem_4_Spring_23\Research\report2\data_transfer\simulate_minibatch_dataload\pubmed


"""

fig, ax = plt.subplots(1,1, figsize=(5, 3))
# fig.suptitle('Time to accuracy')

ax.plot(n_gpus, pubmed, label='Pubmed', marker='+', color='g')
ax.plot(n_gpus, arxiv, label='Ogbn-rxiv', marker='+', color='b')
ax.plot(n_gpus, reddit, label='Reddit', marker='+', color="orange")
ax.plot(n_gpus, products, label='Ogbn-products', marker='+', color='r')



# for a, col in zip(ax, cols):
#     a.set_title(col, size=17)
# for i in range(4):
    # for j in range(4):
    # ax[i, j].tick_params(axis='both', which='major', labelsize=6)
    # ax[i].tick_params(axis='both', labelsize=12)


# n_gpus = [1, 2, 3, 4]

from matplotlib.ticker import MultipleLocator

# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
# fig.subplots_adjust(hspace=0.05)  # adjust space between Axes



fig.supxlabel('Num Partitions', size=20)
fig.supylabel('FG/MB ratio', size=20)
# cols = ["Pubmed", "Ogbn-arxiv", "Reddit", "Ogbn-products"]



# ax1.plot(n_gpus, pubmed, label='Pubmed', marker='+', color='g')
# ax1.plot(n_gpus, arxiv, label='Ogbn-rxiv', marker='+', color='b')
# ax1.plot(n_gpus, reddit, label='Reddit', marker='+', color="orange")
# ax2.plot(n_gpus, products, label='Ogbn-products', marker='+', color='r')



# # zoom-in / limit the view to different portions of the data
# ax1.set_ylim(3.0, 10.0)  # outliers only
# ax2.set_ylim(0, 0.5)  # most of the data

# # hide the spines between ax and ax2
# ax1.spines.bottom.set_visible(False)
# ax2.spines.top.set_visible(False)
# ax1.xaxis.tick_top()
# ax1.tick_params(labeltop=False)  # don't put tick labels at the top
# ax2.xaxis.tick_bottom()

# # Now, let's turn towards the cut-out slanted lines.
# # We create line objects in axes coordinates, in which (0,0), (0,1),
# # (1,0), and (1,1) are the four corners of the Axes.
# # The slanted lines themselves are markers at those locations, such that the
# # lines keep their angle and position, independent of the Axes size or scale
# # Finally, we need to disable clipping.

# d = .5  # proportion of vertical to horizontal extent of the slanted line
# kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
#               linestyle="none", color='k', mec='k', mew=1, clip_on=False)
# ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
# ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)


# lines = [] 
# labels = [] 
  
# for ax in fig.axes: 
#     Line, Label = ax.get_legend_handles_labels() 
#     # print(Label) 
#     lines.extend(Line) 
#     labels.extend(Label)

# fig.legend(lines, labels, loc='center right') 

plt.legend(fontsize=10)
plt.tight_layout()
plt.show()

"""