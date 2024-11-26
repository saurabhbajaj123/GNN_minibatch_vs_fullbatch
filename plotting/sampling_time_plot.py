
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_area_auto_adjustable
import numpy as np
import math


n_gpus = [2,3,4,8,16,32]

pubmed_df = pd.read_csv(r"result\sampling_time\pubmed_time.csv")
arxiv_df = pd.read_csv(r"result\sampling_time\arxiv_time.csv")
reddit_df = pd.read_csv(r"result\sampling_time\reddit_time.csv")
products_df = pd.read_csv(r"result\sampling_time\products_time.csv")
papers_df = pd.read_csv(r"result\sampling_time\papers_time.csv")


pubmed_df['rest'] = pubmed_df['Forward'] + pubmed_df['Backward']
arxiv_df['rest'] = arxiv_df['Forward'] + arxiv_df['Backward']
reddit_df['rest'] = reddit_df['Forward'] + reddit_df['Backward']
products_df['rest'] = products_df['Forward'] + products_df['Backward']
papers_df['rest'] = papers_df['Forward'] + papers_df['Backward']

pubmed_df['ratio'] = pubmed_df['Sampling']/pubmed_df['rest']
arxiv_df['ratio'] = arxiv_df['Sampling']/arxiv_df['rest']
reddit_df['ratio'] = reddit_df['Sampling']/reddit_df['rest']
products_df['ratio'] = products_df['Sampling']/products_df['rest']
papers_df['ratio'] = papers_df['Sampling']/papers_df['rest']

pubmed = pubmed_df['ratio'].tolist()
arxiv = arxiv_df['ratio'].tolist()
reddit = reddit_df['ratio'].tolist()
products = products_df['ratio'].tolist()
papers = papers_df['ratio'].tolist()

print(pubmed)
print(arxiv)
print(reddit)
print(products)
print(papers)

fig, ax = plt.subplots(1,1, figsize=(5, 3))
# fig.suptitle('Time to accuracy')
fig.supxlabel('Num Partitions', size=15)
fig.supylabel('Sampling v/s rest', size=15)
cols = ["Pubmed", "Ogbn-arxiv", "Reddit", "Ogbn-products", 'Ogbn-papers100M']

ax.plot(n_gpus, pubmed, label='Pubmed', marker='+',)
ax.plot(n_gpus, arxiv, label='Ogbn-arxiv', marker='+')
ax.plot(n_gpus, reddit, label='Reddit', marker='+')
ax.plot(n_gpus, products, label='Ogbn-products', marker='+')
ax.plot(n_gpus, papers, label='Ogbn-papers100m', marker='+')

ax.tick_params(axis='both', which='major', labelsize=12)
# ax.set_yscale('exp')
# plt.legend(fontsize =15)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# fig.legend(loc='upper left', fancybox=True, ncol=1,  fontsize=18, bbox_to_anchor=(1.05, 1.0))
plt.tight_layout()
plt.show()
