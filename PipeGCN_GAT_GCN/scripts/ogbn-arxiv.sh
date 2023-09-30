#!/bin/bash

#SBATCH --job-name pipgcn-gat-ogbn-arxiv   ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --mem=50GB  # memory per CPU core
#SBATCH --time=0-20:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --constraint=m40
#SBATCH --output=result_arxiv.txt

source /work/sbajaj_umass_edu/GNNEnv/bin/activate

python main.py \
  --dataset ogbn-arxiv \
  --dropout 0.3 \
  --lr 0.003 \
  --n-partitions 4 \
  --n-epochs 500 \
  --model gat \
  --num-heads 2 \
  --n-layers 3 \
  --n-hidden 256 \
  --log-every 5 \
  --use-pp
  # --enable-pipeline \
