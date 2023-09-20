#!/bin/bash

#SBATCH --job-name pipgcn-gat-pubmed   ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --mem=20GB  # memory per CPU core
#SBATCH --time=0-02:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --constraint=m40
#SBATCH --output=result_pubmed.txt

source /work/sbajaj_umass_edu/GNNEnv/bin/activate

python main.py \
  --dataset pubmed \
  --dropout 0.5 \
  --lr 0.0001 \
  --n-partitions 4 \
  --n-epochs 1000 \
  --model gat \
  --num-heads 2 \
  --n-layers 2 \
  --n-hidden 128 \
  --log-every 5 \
  --use-pp \
  # --enable-pipeline \
  # --convergence-threshold 1e-8 \
  # --norm layer\
  # --seed 837330801 \
  # --fix-seed \
  # --inductive \
  # --parts-per-node 2 \
  # --backend nccl \
