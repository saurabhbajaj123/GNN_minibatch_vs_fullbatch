#!/bin/bash

#SBATCH --job-name quiv-red   ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --mem=20GB  # memory per CPU core
#SBATCH --time=0-24:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --nodes=1
#SBATCH --partition=gypsum-m40

cd /home/ubuntu/gnn_mini_vs_full/GNN_minibatch_vs_fullbatch/pytorch_geometric/torch-quiver
source /home/ubuntu/gnn_mini_vs_full/pygenv1/bin/activate


module load cuda/11.8.0
module load gcc/11.2.0
module load uri/main
module load NCCL/2.12.12-GCCcore-11.3.0-CUDA-11.7.0

QUIVER_ENABLE_CUDA=1 python setup.py install

python3 examples/multi_gpu/pyg/pubmed/dist_sampling_ogb_pubmed_quiver.py
