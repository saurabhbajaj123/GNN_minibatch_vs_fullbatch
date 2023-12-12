#!/bin/bash

#SBATCH --job-name arxiv   ## name that will show up in the queue
#SBATCH --gpus=2
#SBATCH --mem=80GB  # memory per CPU core
#SBATCH --time=0-04:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --partition=gypsum-1080ti
#SBATCH --nodes=1

source /work/sbajaj_umass_edu/GNNEnv/bin/activate

python main.py \
  --dataset ogbn-arxiv \
  --dataset-subgraph-path /work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/DGL/DGL_reference_implementation/subgraph/ogbn-arxiv_frac_0.01_hops_2_subgraph_no_isolated.bin \
  --dropout 0.3 \
  --lr 0.003 \
  --n-partitions 2 \
  --n-epochs 1000 \
  --model graphsage \
  --n-layers 2 \
  --n-hidden 32 \
  --log-every 5 \
  --patience 50 \
  --fix-seed \
  --seed 1261325436 \
  # --enable-pipeline \


# python main.py \
#   --dataset ogbn-arxiv \
#   --dropout 0.3 \
#   --lr 0.003 \
#   --n-partitions 1 \
#   --n-epochs 10 \
#   --model graphsage \
#   --n-layers 8 \
#   --n-hidden 1024 \
#   --log-every 5 \
#   --patience 50 \
#   --fix-seed \
#   --seed 1261325436 \
#   # --enable-pipeline \