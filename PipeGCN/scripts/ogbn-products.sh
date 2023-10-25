#!/bin/bash

#SBATCH --job-name products   ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --mem=150GB  # memory per CPU core
#SBATCH --time=0-20:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --constraint=m40
#SBATCH --nodes=1


source /work/sbajaj_umass_edu/GNNEnv/bin/activate

python main.py \
  --dataset ogbn-products \
  --dropout 0.3 \
  --lr 0.003 \
  --n-partitions 1 \
  --n-epochs 10 \
  --model graphsage \
  --n-layers 5 \
  --n-hidden 128 \
  --log-every 5 \
  --use-pp \
  --fix-seed \
  --patience 100 \
  --enable-pipeline \
  # --seed 837330801 \
  # --norm layer\
  # --inductive \
  # --parts-per-node 2 \
  # --backend nccl \
  # --convergence-threshold 0.0001\
  # SBATCH --constraint=a100

# python main.py \
#   --dataset ogbn-products \
#   --dropout 0.3 \
#   --lr 0.003 \
#   --n-partitions 4 \
#   --n-epochs 500 \
#   --model graphsage \
#   --n-layers 4 \
#   --n-hidden 127 \
#   --log-every 100 \
#   --patience 500 \
#   --use-pp \
#   --fix-seed \
#   --seed 837330801 \
#   # --enable-pipeline \

