#!/bin/bash

#SBATCH --job-name reddit   ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --mem=100GB  # memory per CPU core
#SBATCH --time=0-24:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --partition=gypsum-m40
#SBATCH --nodes=1

source /work/sbajaj_umass_edu/GNNEnv/bin/activate

python main.py \
  --dataset reddit \
  --dropout 0.3 \
  --lr 0.001 \
  --n-partitions 4 \
  --n-epochs 10 \
  --model gcn \
  --n-layers 6 \
  --n-hidden 1024 \
  --num-heads 2 \
  --log-every 5 \
  --patience 50 \
  --fix-seed \
  --use-pp \
  --seed 42 \
  # --enable-pipeline \
  # --inductive \


# python main.py \
#   --dataset reddit \
#   --dropout 0.3 \
#   --lr 0.001 \
#   --n-partitions 4 \
#   --n-epochs 10 \
#   --model gat \
#   --n-layers 4 \
#   --n-hidden 512 \
#   --num-heads 2 \
#   --log-every 5 \
#   --patience 50 \
#   --fix-seed \
#   --use-pp \
#   --seed 42 \
#   --enable-pipeline \