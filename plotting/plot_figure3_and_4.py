import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_area_auto_adjustable
import numpy as np

fig, ax = plt.subplots(1,3, figsize=(10,4))
# fig.suptitle('Time to accuracy')
fig.supxlabel('Time (s)', fontsize=20)
fig.supylabel('Test accuracy', fontsize=20)
cols = ["Ogbn-arxiv", "Reddit", "Ogbn-products"]

for a, col in zip(ax, cols):
    a.set_title(col, fontsize=20)
for i in range(3):
    # for j in range(4):
    # ax[i, j].tick_params(axis='both', which='major', labelsize=6)
    ax[i].tick_params(axis='both', which='minor', labelsize=7)
    ax[i].tick_params(axis='both', which='major', labelsize=15)

for i, dataset in enumerate(["arxiv", "reddit", "products"]):
    for j, system in enumerate(["full_graph", "pipegcn", "bns_gcn", "adaqp", "dgl", "quiver"]):
        test_acc_df[dataset][system] = pd.read_csv(f"results/{dataset}/test_acc_{dataset}_{system}.csv")

test_acc_df = defaultdict(defaultdict(pandas.DataFrame, None), None)


for i, dataset in enumerate(["arxiv", "reddit", "products"]):
    for j, system in enumerate(["full_graph", "pipegcn", "bns_gcn", "adaqp", "dgl", "quiver"]):
        ax[i].plot( test_acc_df[dataset][system]['train_time'],  test_acc_df[dataset][system]['test_acc'], label=system, marker='+')

fig.legend(loc='upper center', fancybox=True, ncol=3,  fontsize=17, bbox_to_anchor=(0.5, 1.0))
plt.tight_layout()
plt.show()
