#!/bin/bash

#SBATCH --job-name arxiv   ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --mem=50GB  # memory per CPU core
#SBATCH --time=0-24:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --partition=gypsum-m40
#SBATCH --nodes=1


source /work/sbajaj_umass_edu/GNNEnv/bin/activate

python saint_main.py \
  --dataset ogbn-arxiv \
  --model graphsage \
  --sampling saint \
  --dropout 0.3 \
  --lr 0.001 \
  --n-epochs 10000 \
  --n-gpus 4 \
  --n-layers 4 \
  --n-hidden 1024 \
  --batch-size 4096 \
  --budget_node_edge 6000 \
  --budget_rw_0 256 \
  --budget_rw_1 16 \
  --mode_saint node \
  --num-iters 1000 \
  --log-every 50 \
  # --seed 6260732369359939000 \