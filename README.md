# GNN_minibatch_vs_fullbatch
Compare training accuracies of full batch training with mini batch training

This repository contains all the code which was used for the Graph Neural Network Training Systems: **A Comparison of
Full-Graph and Mini-Batch. [Experiment, Analysis & Benchmark] paper.**

Here is the extended version of the report that was submitted to VLDB April 1st 2025

[VLDB25_GNN_benchmarking.pdf](https://github.com/user-attachments/files/15538780/VLDB24_GNN_benchmarking_June2024_2025submission.pdf)

Please go to the official repositories of the systems for updated code:
- PipeGCN - https://github.com/GATECH-EIC/PipeGCN
- BNS-GCN - https://github.com/GATECH-EIC/BNS-GCN
- Quiver - https://github.com/quiver-team/torch-quiver
- DGL - https://github.com/dmlc/dgl/tree/master/examples/multigpu (https://github.com/dmlc/dgl/tree/master/examples/pytorch)
- DistDGL - https://github.com/dmlc/dgl/tree/master/examples/distributed

### Environment

#### Hardware Dependencies

- Single host experiments: 4 GPUs
- Multi-host experiments: 3 hosts with 4 GPUs each
- Lower end experiments:
    - m40 GPUs with 24GB memory
    - 2 Intel Xeon E5-2620 v3 CPUs with 12 cores each, 256GB of host memory
- Higher end experiments: 
    - a100 GPUs with 80GB memory
    - 2 Intel Xeon Platinum 8480+ CPUs with 56 cores each, 256GB of host memory


#### Software Dependencies

- Ubuntu 18.04
- Python 3.9.12
- CUDA 11.8
- [PyTorch 2.0.1](https://github.com/pytorch/pytorch)
- [DGL 1.1.2+cu118](https://github.com/chwan-rice/dgl)
- [OGB 1.3.6](https://ogb.stanford.edu/docs/home/)


### Installation

#### PipeGCN
Follow the steps on the PipeGCN repository [https://github.com/GATECH-EIC/PipeGCN]
```
git clone https://github.com/GATECH-EIC/PipeGCN.git
```
#### BNS-GCN
```
git clone https://github.com/GATECH-EIC/BNS-GCN.git
```

#### Quiver
```
git clone https://github.com/quiver-team/torch-quiver.git && cd torch-quiver
QUIVER_ENABLE_CUDA=1 python setup.py install
```

#### DistDGL

#### AdaQP
Follow the steps in the repository () to install AdaQP

### Datasets
We run our experiments on the following datasets:
- [ogbn datasets](https://ogb.stanford.edu/docs/nodeprop/)
    - ogbn-arxiv
    - ogbn-products
    - ogbn-papers100M
- [Reddit](https://snap.stanford.edu/graphsage/#datasets)
- [Pubmed](https://linqs.org/datasets/#pubmed-diabetes)
- [Orkut](https://snap.stanford.edu/data/com-Orkut.html)


### Experiments

#### Figure 3
This figure shows the test accuracy for different datasets and different systems. To reproduce this figure, you need to run the training script for all the systems and datasets. Save the test accuracy in a csv file and plot it using the plot_figure3.py script.

Follow these steps to run the training script for each <dataset> and <system>:
<system> = {PipeGCN, BNS-GCN, DGL/DGL_reference_implementation/Distributed/MultiGPU, pytorch_geometric/quiver1}
<dataset> = {reddit, ogbn-arxiv, ogbn-products, ogbn-papers100M, pubmed, orkut}
```
cd <system>/
bash scripts/<dataset>.sh
```
The bash script arguments can be changed according to the desired experimental settings.

Save the test accuracies in a csv file and then plot the figure using the plotting/plot_figure3.py script.


