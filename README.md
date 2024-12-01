# GNN_minibatch_vs_fullbatch
Compare training accuracies of full batch training with mini batch training

This repository contains all the code which was used for the Graph Neural Network Training Systems: **A Comparison of
Full-Graph and Mini-Batch. [Experiment, Analysis & Benchmark] paper.**

Here is the extended version of the report that was submitted to VLDB April 1st 2025

[VLDB25_GNN_benchmarking.pdf](https://github.com/saurabhbajaj123/GNN_minibatch_vs_fullbatch/blob/main/VLDB24_GNN_benchmarking_Dec1.pdf)

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
Follow the steps in the repository [GNN_MB-FB_Comparison fb-training](https://github.com/goodluck-hojae/GNN_MB-FB_Comparison/tree/main/fb-training) to install and run AdaQP

#### Cluster-GCN and SAINT sampler 
Follow the steps in the repository [GNN_MB-FB_Comparison mb-training](https://github.com/goodluck-hojae/GNN_MB-FB_Comparison/tree/main/mb-training)


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
To reproduce the figures and tables, you need to run the training script for all the systems and datasets. 

Follow these steps to run the training script for each *dataset* and *system*:
*system* = {PipeGCN, BNS-GCN, DGL/DGL_reference_implementation/Distributed/MultiGPU, pytorch_geometric/quiver1}.
*dataset* = {reddit, ogbn-arxiv, ogbn-products, ogbn-papers100M, pubmed, orkut}
```
cd <system>/
bash scripts/<dataset>.sh
```

#### Table 3/4
Train the model till convergence (i.e. when the val accuracies do not improve), 
`total_time` at convergence is the `TTA`, and the `ET` is the ratio of `total_time` and `number of epochs`

#### Table 5/6
Run the training scripts with different number of GPUs/partitions, and obtain the `TTA` and `ET` for each GPU/partition count.
Speed-ups are obtaained by taking a ratio of the `TTA` with the `TTA` for 1 GPU/partition.

#### Table 7
Run hyperparemeter search for full-graph (PipeGCN-vanilla), obtain the best set of parameters (which give the highest accuracy), this gives the FG-train - FG-search. Use these parameters to run mini-batch (DGL) FG-search - MB-train.

Do hyperparameter search for mini-batch (MB-search - MB-train), use these parameters to train using full-graph (MB-search - FG-train) accuracy numbers.


#### Table 8
Run the training scripts with the best hyperparameters for all the datasets and systems. For mini-batch (DGL/DGL_reference_implementation/Distributed/MultiGPU), run with different sampling algorithms from [GNN_MB-FB_Comparison mb-training](https://github.com/goodluck-hojae/GNN_MB-FB_Comparison/tree/main/mb-training).

#### Figure 3
This figure shows the test accuracy for different datasets and different systems. To reproduce this figure, you need to run the training script for all the systems and datasets. Save the test accuracy in a csv file and plot it using the plot_figure3.py script.

The bash script arguments can be changed according to the desired experimental settings.

Save the test accuracies in a csv file and then plot the figure using the plotting/plot_figure3_and_4.py script.
```
python plotting/plot_figure3_and_4.py
```

#### Figure 7
This figure shows the FG/MB ratio of communication cost. 
To calculate the data transfer in MB, follow this repository [simulate_minibatch_dataload](https://github.com/juelinl/pebble).
To calculate the data transfer in FG, we use the metis_partition_assignment function in DGL to obtain the vertex transfered out of each GPU and then calculate the data transfer by summing up the number of edges transfered out of each GPU. 
To obtain the vertex transfered out of each GPU, we use the following script:
```
python PipeGCN/cross_gpu_data_transfer.py
```
To reproduce the figure, run the following command:
```
python plotting/data_transfer_plot.py
```

#### Figure 8
This figure shows the ratio of the computation cost of FG (PipeGCN - vanilla) and MB (DGL) training for GraphSAGE, GCN and GAT. To reproduce the figure run the training script for different number of partitions/GPUs/workers and save the floating point operations in a csv file in result/compute_cost/ folder.
Then run the following command to plot the figure:
```
python plotting/flops_plot.py
```

#### Figure 9
Sampling time vs training time comparison
Run the training script for different number of GPUs and save the sampling time, forward pass and backward pass times in a csv file in result/sampling_time/ folder.
And run the following command to plot the figure
```
python plotting/sampling_time_plot.py
```

