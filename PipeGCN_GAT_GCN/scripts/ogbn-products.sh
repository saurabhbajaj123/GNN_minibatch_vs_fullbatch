#!/bin/bash

#SBATCH --job-name products   ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --mem=100GB  # memory per CPU core
#SBATCH --time=0-24:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --constraint=m40
#SBATCH --nodes=1


source /work/sbajaj_umass_edu/GNNEnv/bin/activate

# python main.py \
#   --dataset ogbn-products \
#   --dropout 0.3 \
#   --lr 0.002 \
#   --n-partitions 4 \
#   --n-epochs 10 \
#   --model gcn \
#   --num-heads 2 \
#   --n-layers 5 \
#   --n-hidden 128 \
#   --log-every 5 \
#   --patience 100 \
#   --fix-seed \
#   --seed 42 \
#   # --enable-pipeline \
#   # --use-pp \
#   # --norm layer\
#   # --inductive \
#   # --parts-per-node 2 \
#   # --backend nccl \
#   # --convergence-threshold 0.0001\
#   # SBATCH --constraint=a100


  python main.py \
  --dataset ogbn-products \
  --dropout 0.3 \
  --lr 0.002 \
  --n-partitions 4 \
  --n-epochs 10 \
  --model gcn \
  --num-heads 2 \
  --n-layers 4 \
  --n-hidden 256 \
  --log-every 5 \
  --patience 100 \
  --fix-seed \
  --seed 42 \
  --enable-pipeline \
