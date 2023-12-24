#!/bin/bash

#SBATCH --job-name products   ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --mem=50GB  # memory per CPU core
#SBATCH --time=0-24:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --partition=gypsum-m40
#SBATCH --nodes=1


source /home/ubuntu/gnn_mini_vs_full/GNNEnv/bin/activate

python main.py \
  --dataset ogbn-products \
  --dataset-subgraph-path /home/ubuntu/gnn_mini_vs_full/GNN_minibatch_vs_fullbatch/DGL/DGL_reference_implementation/subgraph/ogbn-products_frac_0.01_hops_2_subgraph.bin \
  --dropout 0.3 \
  --lr 0.003 \
  --n-partitions 4 \
  --n-epochs 20 \
  --model graphsage \
  --n-layers 2 \
  --n-hidden 32 \
  --log-every 5 \
  --use-pp \
  --fix-seed \
  --patience 100 \
  --seed 837330801 \
  # --enable-pipeline \
  # --norm layer\
  # --inductive \
  # --parts-per-node 2 \
  # --backend nccl \
  # --convergence-threshold 0.0001\
  # SBATCH --constraint=a100

# python main.py \
#   --dataset ogbn-products \
#   --dropout 0.3 \
#   --lr 0.003 \
#   --n-partitions 4 \
#   --n-epochs 500 \
#   --model graphsage \
#   --n-layers 4 \
#   --n-hidden 127 \
#   --log-every 100 \
#   --patience 500 \
#   --use-pp \
#   --fix-seed \
#   --seed 837330801 \
#   # --enable-pipeline \

