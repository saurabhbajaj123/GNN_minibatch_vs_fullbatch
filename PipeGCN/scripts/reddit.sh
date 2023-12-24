#!/bin/bash

#SBATCH --job-name reddit   ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --mem=150GB  # memory per CPU core
#SBATCH --time=0-24:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --partition=gypsum-m40
#SBATCH --nodes=1

source /home/ubuntu/gnn_mini_vs_full/GNNEnv/bin/activate

# python main.py \
#   --dataset reddit \
#   --dropout 0.3 \
#   --lr 0.001 \
#   --n-partitions 4 \
#   --n-epochs 500 \
#   --model graphsage \
#   --n-layers 4 \
#   --n-hidden 1024 \
#   --log-every 10 \
#   --patience 100 \
#   --seed 1122320811 \
#   --enable-pipeline \
#   # --fix-seed \


for n_parts in 4
do
  echo $n_parts
  python main.py \
    --dataset reddit \
    --dropout 0.3 \
    --lr 0.01 \
    --n-partitions $n_parts \
    --n-epochs 5 \
    --model graphsage \
    --n-layers 4 \
    --n-hidden 512 \
    --log-every 10 \
    --patience 100 \
    --seed 1122320811
done