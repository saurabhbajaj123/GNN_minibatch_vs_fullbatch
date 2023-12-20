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
  --lr 0.01 \
  --n-partitions 4 \
  --n-epochs 500 \
  --model graphsage \
  --n-layers 2 \
  --n-hidden 512 \
  --log-every 5 \
  --patience 50 \
  --fix-seed \
  --seed 1261325436 \
  # --enable-pipeline \


  python main.py \
  --dataset ogbn-arxiv \
  --dropout 0.3 \
  --lr 0.01 \
  --n-partitions 4 \
  --n-epochs 500 \
  --model graphsage \
  --n-layers 2 \
  --n-hidden 512 \
  --log-every 5 \
  --patience 50 \
  --fix-seed \
  --seed 1261325436 \
  --enable-pipeline \
