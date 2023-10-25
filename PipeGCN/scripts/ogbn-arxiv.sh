#!/bin/bash

#SBATCH --job-name arxiv   ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --mem=80GB  # memory per CPU core
#SBATCH --time=0-24:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --partition=gypsum-m40
#SBATCH --nodes=1

source /work/sbajaj_umass_edu/GNNEnv/bin/activate

python main.py \
  --dataset ogbn-arxiv \
  --dropout 0.3 \
  --lr 0.003 \
  --n-partitions 12 \
  --n-epochs 100 \
  --model graphsage \
  --n-layers 3 \
  --n-hidden 256 \
  --log-every 5 \
  --patience 50 \
  --fix-seed \
  --seed 1261325436 \
  --enable-pipeline \


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