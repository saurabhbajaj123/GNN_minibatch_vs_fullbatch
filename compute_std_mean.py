import numpy as np

pipegcn_FB_FB_pubmed = [0.766, 0.766, 0.766, 0.766, 0.765]

pipegcn_FB_FB_reddit = [0.9689, 0.9689, 0.9691, 0.9688, 0.969]

pipegcn_FB_FB_products = [0.7824, 0.7842, 0.7815, 0.7839, 0.7802]

pipegcn_FB_FB_arxiv = [0.7197, 0.7176, 0.7218, 0.7217, 0.719]
# Calculate the standard deviation using numpy
std_dev = np.std(pipegcn_FB_FB_pubmed)




# print(np.std(pipegcn_FB_FB_pubmed))
# print(np.std(pipegcn_FB_FB_arxiv))
# print(np.std(pipegcn_FB_FB_reddit))
# print(np.std(pipegcn_FB_FB_products))



# FB archi, MB training
pipegcn_FB_MB_pubmed = [0.777, 0.77, 0.774, 0.775, 0.766]
pipegcn_FB_MB_arxiv = [0.7066, 0.711, 0.7014, 0.7103]
pipegcn_FB_MB_reddit = [0.9654, 0.9689, 0.9691, 0.9688, 0.969] # 10113534
pipegcn_FB_MB_products = [0.7879, 0.7865]



# print(np.mean(pipegcn_FB_MB_pubmed))
# print(np.std(pipegcn_FB_MB_pubmed))
# print(np.mean(pipegcn_FB_MB_arxiv))
# print(np.std(pipegcn_FB_MB_arxiv))
# print(np.mean(pipegcn_FB_MB_reddit))
# print(np.std(pipegcn_FB_MB_reddit))
# print(np.mean(pipegcn_FB_MB_products))
# print(np.std(pipegcn_FB_MB_products))



# MB archi, FB training
pipegcn_MB_FB_pubmed = [0.523, 0.489, 0.557, 0.506]
pipegcn_MB_FB_arxiv = [0.3467, 0.221, 0.2156, 0.3976]
pipegcn_MB_FB_reddit = [0.5016, 0.4627, 0.431, 0.4043]
pipegcn_MB_FB_products = [0.7695, 0.7668, 0.7617]

# print(np.mean(pipegcn_MB_FB_pubmed))
# print(np.std(pipegcn_MB_FB_pubmed))
# print(np.mean(pipegcn_MB_FB_arxiv))
# print(np.std(pipegcn_MB_FB_arxiv))
# print(np.mean(pipegcn_MB_FB_reddit))
# print(np.std(pipegcn_MB_FB_reddit))
print(np.mean(pipegcn_MB_FB_products))
print(np.std(pipegcn_MB_FB_products))


# MB archi, MB training
pipegcn_MB_MB_pubmed = [0.764, 0.77, 0.763, 0.762, 0.769]


# print(np.mean(pipegcn_MB_MB_pubmed))
# print(np.std(pipegcn_MB_MB_pubmed))