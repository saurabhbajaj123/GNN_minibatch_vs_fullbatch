#!/bin/bash

#SBATCH --job-name reddit   ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --mem=50GB  # memory per CPU core
#SBATCH --time=0-08:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --constraint=m40
#SBATCH --nodes=1

source /work/sbajaj_umass_edu/GNNEnv/bin/activate

python main.py \
  --dataset reddit \
  --dropout 0.5 \
  --lr 0.001 \
  --n-partitions 4 \
  --n-epochs 1000 \
  --model graphsage \
  --n-layers 7 \
  --n-hidden 256 \
  --log-every 5 \
  --patience 50 \
  --seed 42 \
  # --enable-pipeline \
  # --fix-seed \

# python main.py \
#   --dataset reddit \
#   --dropout 0.3 \
#   --lr 0.001 \
#   --n-partitions 4 \
#   --n-epochs 1000 \
#   --model graphsage \
#   --n-layers 4 \
#   --n-hidden 312 \
#   --log-every 99 \
#   --fix-seed \
#   --patience 500 \
#   --seed 1586505639 \
#   --use-pp \
#   # --enable-pipeline \
#   # --inductive \