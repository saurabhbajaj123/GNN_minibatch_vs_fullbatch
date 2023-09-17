#!/bin/bash

#SBATCH --job-name test   ## name that will show up in the queue
#SBATCH -p gpu-preempt
#SBATCH --gpus=4
#SBATCH --mem=20GB  # memory per CPU core
#SBATCH --time=0-00:10:00  ## time for analysis (day-hour:min:sec)
#SBATCH --constraint=a100
#SBATCH --output=result.txt

source /work/sbajaj_umass_edu/GNNEnv/bin/activate

python main.py \
  --dataset pubmed \
  --dropout 0.5 \
  --lr 0.0001 \
  --n-partitions 4 \
  --n-epochs 2000 \
  --model graphsage \
  --n-layers 3 \
  --n-hidden 187 \
  --log-every 5 \
  --use-pp \
  --convergence-threshold 1e-4 \
  --enable-pipeline \
  # --norm layer\
  # --seed 837330801 \
  # --fix-seed \
  # --inductive \
  # --parts-per-node 2 \
  # --backend nccl \
