#!/bin/bash

#SBATCH --job-name arxiv   ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --mem=80GB  # memory per CPU core
#SBATCH --time=0-24:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --partition=gypsum-m40
#SBATCH --nodes=1

source /home/ubuntu/gnn_mini_vs_full/GNNEnv/bin/activate


for n_parts in 1
do
  echo $n_parts
  python main.py \
    --dataset ogbn-arxiv \
    --dropout 0.3 \
    --lr 0.01 \
    --n-partitions 4 \
    --n-epochs 5 \
    --model graphsage \
    --n-layers 6 \
    --n-hidden 1024 \
    --log-every 10 \
    --patience 50 \
    --fix-seed \
    --seed 1261325436
done

for n_parts in 1
do
  echo $n_parts
  python main.py \
    --dataset ogbn-arxiv \
    --dropout 0.3 \
    --lr 0.01 \
    --n-partitions $n_parts \
    --n-epochs 5 \
    --model graphsage \
    --n-layers 6 \
    --n-hidden 1024 \
    --log-every 10 \
    --patience 50 \
    --fix-seed \
    --seed 1261325436 \
    --enable-pipeline
done




