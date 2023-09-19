#!/bin/bash

#SBATCH --job-name arxiv   ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --mem=20GB  # memory per CPU core
#SBATCH --time=0-02:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --output=result_arxiv.txt

source /work/sbajaj_umass_edu/GNNEnv/bin/activate

python main.py \
  --dataset ogbn-arxiv \
  --dropout 0.3 \
  --lr 0.005 \
  --n-partitions 4 \
  --n-epochs 1000 \
  --model graphsage \
  --n-layers 9 \
  --n-hidden 256 \
  --log-every 5 \
  --patience 50 \
  --fix-seed \
  # --enable-pipeline \
  # --seed 1261325436 \


# python main.py \
#   --dataset ogbn-arxiv \
#   --dropout 0.3 \
#   --lr 0.003 \
#   --n-partitions 4 \
#   --n-epochs 1000 \
#   --model graphsage \
#   --n-layers 3 \
#   --n-hidden 256 \
#   --log-every 5 \
#   --enable-pipeline \
#   --use-pp
