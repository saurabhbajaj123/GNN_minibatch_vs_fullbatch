#!/bin/bash

#SBATCH --job-name arxiv  ## name that will show up in the queue
#SBATCH --partition=cpu
#SBATCH -c 64
#SBATCH --mem=250GB  # memory per CPU core
#SBATCH --time=0-24:00:00  ## time for analysis (day-hour:min:sec)


source /work/sbajaj_umass_edu/GNNEnv/bin/activate

python train.py \
  --dataset ogbn-arxiv \
  --model SAGE \
  --dropout 0.3 \
  --lr 0.003 \
  --n-epochs 100 \
  --n-layers 128 \
  --n-hidden 64 \
  --batch-size 1024 \
  --fanout 4 \
  --num-heads 7 \
  --agg mean \
  --log-every 5 \
  --patience 50 \
  --seed 42 \
  # --seed 3485963027166655500 \