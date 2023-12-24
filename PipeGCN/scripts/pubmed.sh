#!/bin/bash

#SBATCH --job-name pubmed   ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --mem=20GB  # memory per CPU core
#SBATCH --time=0-05:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --partition=gypsum-m40
#SBATCH --nodes=1

source /home/ubuntu/gnn_mini_vs_full/GNNEnv/bin/activate

python main.py \
  --dataset pubmed \
  --dropout 0.7 \
  --lr 0.001 \
  --n-partitions 1 \
  --n-epochs 5 \
  --model graphsage \
  --n-layers 3 \
  --n-hidden 256 \
  --log-every 10 \
  --use-pp \
  --patience 50 \
  --fix-seed \
  --seed 1344439319 \
  --enable-pipeline \


