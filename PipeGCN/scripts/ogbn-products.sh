#!/bin/bash

#SBATCH --job-name ogbn-products   ## name that will show up in the queue
#SBATCH -p gpu-preempt
#SBATCH --gpus=4
#SBATCH --mem=50GB  # memory per CPU core
#SBATCH --time=0-00:30:00  ## time for analysis (day-hour:min:sec)
#SBATCH --output=result.txt

source /work/sbajaj_umass_edu/GNNEnv/bin/activate

python main.py \
  --dataset ogbn-products \
  --dropout 0.3 \
  --lr 0.003 \
  --n-partitions 4 \
  --n-epochs 500 \
  --model graphsage \
  --n-layers 4 \
  --n-hidden 127 \
  --log-every 5 \
  --use-pp \
  --fix-seed \
  --seed 837330801 \
  # --enable-pipeline \
  # --norm layer\
  # --inductive \
  # --parts-per-node 2 \
  # --backend nccl \
  # --convergence-threshold 0.0001\
  # SBATCH --constraint=a100
