#!/bin/bash

#SBATCH --job-name pubmed   ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --mem=20GB  # memory per CPU core
#SBATCH --time=0-05:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --output=result_pubmed.txt
#SBATCH --constraint=m40
#SBATCH --nodes=1

source /work/sbajaj_umass_edu/GNNEnv/bin/activate

python main.py \
  --dataset pubmed \
  --dropout 0.7 \
  --lr 0.0001 \
  --n-partitions 4 \
  --n-epochs 2000 \
  --model graphsage \
  --n-layers 4 \
  --n-hidden 256 \
  --log-every 5 \
  --use-pp \
  --patience 50 \
  --enable-pipeline \
  --fix-seed \
  # --seed 1344439319 \



# python main.py \
#   --dataset pubmed \
#   --dropout 0.7565688403188127 \
#   --lr 0.0001 \
#   --n-partitions 4 \
#   --n-epochs 2000 \
#   --model graphsage \
#   --n-layers 3 \
#   --n-hidden 187 \
#   --log-every 5 \
#   --use-pp \
#   --patience 1000 \
#   --enable-pipeline \
#   --seed 1344439319  \
#   --fix-seed \
#   # --norm layer\
#   # --inductive \
#   # --parts-per-node 2 \
#   # --backend nccl \
