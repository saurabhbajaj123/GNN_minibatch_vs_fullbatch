#!/bin/bash

#SBATCH --job-name arxiv   ## name that will show up in the queue
#SBATCH --gpus=1
#SBATCH --mem=20GB  # memory per CPU core
#SBATCH --time=0-24:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --constraint=m40
#SBATCH --nodes=1

source /work/sbajaj_umass_edu/GNNEnv/bin/activate

python testing_new_saint_class.py \
  --dataset ogbn-arxiv \
  --model SAGE \
  --n-epochs 2000 \
  --n-layers 2 \
  --n-hidden 512 \
  --num-heads 3 \
  --lr 0.002 \
  --dropout 0.3 \
  --seed 42 \
  --num-iters 1000 \
  --budget_node_edge 4000 \
  --budget_rw_0 256 \
  --budget_rw_1 16 \
  --mode_saint node \
  --batch_size 1024 \
  --log_every 50 \