#!/bin/bash

#SBATCH --job-name arxiv   ## name that will show up in the queue.
#SBATCH --gpus=1
#SBATCH --mem=15GB  # memory per CPU core
#SBATCH --time=0-04:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --constraint=m40


source /work/sbajaj_umass_edu/GNNEnv/bin/activate

python3 gcn_train.py \
  --dataset ogbn-arxiv \
  --dropout 0.5 \
  --lr 0.002 \
  --n-epochs 2000 \
  --n-layers 2 \
  --n-hidden 64 \
  --log-every 5 \
  --seed 42 \