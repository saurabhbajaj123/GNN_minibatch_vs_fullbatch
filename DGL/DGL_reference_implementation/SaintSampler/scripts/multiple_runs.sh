#!/bin/bash

#SBATCH --job-name multiple   ## name that will show up in the queue
#SBATCH --gpus=1
#SBATCH --mem=50GB  # memory per CPU core
#SBATCH --time=0-24:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --constraint=m40
#SBATCH --nodes=1

source /work/sbajaj_umass_edu/GNNEnv/bin/activate

# python testing_new_saint_class.py \
#   --dataset ogbn-arxiv \
#   --model GCN \
#   --n-epochs 1000 \
#   --n-layers 2 \
#   --n-hidden 256 \
#   --num-heads 2 \
#   --lr 0.001 \
#   --dropout 0.5 \
#   --seed 42 \
#   --device_id 1 \
#   --budget_node_edge 256 \
#   --budget_rw_0 256 \
#   --budget_rw_1 16 \
#   --mode_saint node \
#   --batch_size 1024 \
#   --log_every 5 \
  # --patience 50 \

python testing_new_saint_class.py \
  --dataset ogbn-products \
  --model GCN \
  --n-epochs 1000 \
  --n-layers 2 \
  --n-hidden 256 \
  --num-heads 2 \
  --lr 0.001 \
  --dropout 0.5 \
  --seed 42 \
  --device_id 1 \
  --num-iters 1000 \
  --budget_node_edge 5000 \
  --budget_rw_0 256 \
  --budget_rw_1 16 \
  --mode_saint node \
  --batch_size 1024 \
  --log_every 5 \
  --patience 50 \

python testing_new_saint_class.py \
  --dataset pubmed \
  --model GCN \
  --n-epochs 1000 \
  --n-layers 2 \
  --n-hidden 256 \
  --num-heads 2 \
  --lr 0.001 \
  --dropout 0.5 \
  --seed 42 \
  --device_id 1 \
  --num-iters 1000 \
  --budget_node_edge 5000 \
  --budget_rw_0 256 \
  --budget_rw_1 16 \
  --mode_saint node \
  --batch_size 1024 \
  --log_every 5 \
  --patience 50 \

python testing_new_saint_class.py \
  --dataset reddit \
  --model GCN \
  --n-epochs 1000 \
  --n-layers 2 \
  --n-hidden 256 \
  --num-heads 2 \
  --lr 0.001 \
  --dropout 0.5 \
  --seed 42 \
  --device_id 1 \
  --num-iters 1000 \
  --budget_node_edge 5000 \
  --budget_rw_0 256 \
  --budget_rw_1 16 \
  --mode_saint node \
  --batch_size 1024 \
  --log_every 5 \
  --patience 50 \
