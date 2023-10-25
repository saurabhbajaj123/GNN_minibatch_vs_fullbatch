#!/bin/bash

#SBATCH --job-name pubmed   ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --mem=15GB  # memory per CPU core
#SBATCH --time=0-024:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --constraint=m40
#SBATCH --nodes=1

source /work/sbajaj_umass_edu/GNNEnv/bin/activate

python main.py \
  --dataset pubmed \
  --dropout 0.5 \
  --lr 0.001 \
  --n-partitions 4 \
  --n-epochs 500 \
  --model gcn \
  --n-layers 6 \
  --n-hidden 1024 \
  --num-heads 1 \
  --log-every 5 \
  --patience 50 \
  --use-pp \
  --seed 42 \
  # --enable-pipeline \
  # --convergence-threshold 1e-8 \
  # --norm layer\
  # --fix-seed \
  # --inductive \
  # --parts-per-node 2 \
  # --backend nccl \
