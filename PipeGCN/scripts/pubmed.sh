#!/bin/bash

#SBATCH --job-name pubmed   ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --mem=20GB  # memory per CPU core
#SBATCH --time=0-08:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --output=result_pubmed.txt

source /work/sbajaj_umass_edu/GNNEnv/bin/activate

python main.py \
  --dataset pubmed \
  --dropout 0.7565688403188127 \
  --lr 0.005 \
  --n-partitions 4 \
  --n-epochs 1000 \
  --model graphsage \
  --n-layers 6 \
  --n-hidden 256 \
  --log-every 5 \
  --use-pp \
  --patience 50 \
  --enable-pipeline \
  --master_addr 10.100.31.54 \
  --port 1234 \
  --node-rank 3 \
  --parts-per-node 1 \
  --fix-seed \
  # --seed 1344439319 \



# python main.py \
#   --dataset pubmed \
#   --dropout 0.5 \
#   --lr 0.0001 \
#   --n-partitions 4 \
#   --n-epochs 2000 \
#   --model graphsage \
#   --n-layers 3 \
#   --n-hidden 187 \
#   --log-every 5 \
#   --use-pp \
#   --convergence-threshold 1e-4 \
#   --enable-pipeline \
#   # --norm layer\
#   # --seed 837330801 \
#   # --fix-seed \
#   # --inductive \
#   # --parts-per-node 2 \
#   # --backend nccl \
