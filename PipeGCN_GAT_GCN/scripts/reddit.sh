#!/bin/bash

#SBATCH --job-name pipgcn-gat-reddit   ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --mem=40GB  # memory per CPU core
#SBATCH --time=0-10:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --constraint=m40
#SBATCH --output=result_reddit.txt


source /work/sbajaj_umass_edu/GNNEnv/bin/activate

python main.py \
  --dataset reddit \
  --dropout 0.3 \
  --lr 0.001 \
  --n-partitions 4 \
  --n-epochs 1000 \
  --model gcn \
  --n-layers 2 \
  --n-hidden 128 \
  --num-heads 3 \
  --log-every 5 \
  --fix-seed \
  --seed 42 \
  --use-pp \
  --enable-pipeline \
  # --inductive \