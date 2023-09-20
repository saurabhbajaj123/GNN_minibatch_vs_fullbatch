#!/bin/bash

#SBATCH --job-name ogbn-products   ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --mem=50GB  # memory per CPU core
#SBATCH --time=0-10:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --constraint=m40
#SBATCH --output=result_products.txt

source /work/sbajaj_umass_edu/GNNEnv/bin/activate

python main.py \
  --dataset ogbn-products \
  --dropout 0.3 \
  --lr 0.003 \
  --n-partitions 4 \
  --n-epochs 100 \
  --model gat \
  --num-heads 2 \
  --n-layers 4 \
  --n-hidden 127 \
  --log-every 5 \
  --fix-seed \
  --seed 42 \
  # --enable-pipeline \
  # --use-pp \
  # --norm layer\
  # --inductive \
  # --parts-per-node 2 \
  # --backend nccl \
  # --convergence-threshold 0.0001\
  # SBATCH --constraint=a100
