#!/bin/bash

#SBATCH --job-name arxiv   ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --mem=50GB  # memory per CPU core
#SBATCH --time=0-24:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --constraint=m40
#SBATCH --nodes=1

source /work/sbajaj_umass_edu/GNNEnv/bin/activate

python main.py \
  --dataset ogbn-arxiv \
  --dropout 0.5 \
  --lr 0.005 \
  --n-partitions 4 \
  --n-epochs 10 \
  --model gat \
  --num-heads 2 \
  --n-layers 3 \
  --n-hidden 1024 \
  --log-every 5 \
  --patience 50 \
  --use-pp \
  --seed 42 \
  --enable-pipeline \



# python main.py \
#   --dataset ogbn-arxiv \
#   --dropout 0.5 \
#   --lr 0.005 \
#   --n-partitions 4 \
#   --n-epochs 10 \
#   --model gat \
#   --num-heads 2 \
#   --n-layers 6 \
#   --n-hidden 1024 \
#   --log-every 5 \
#   --patience 50 \
#   --use-pp \
#   --seed 42 \
#   # --enable-pipeline \