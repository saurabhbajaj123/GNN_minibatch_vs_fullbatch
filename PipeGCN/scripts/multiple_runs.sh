#!/bin/bash

#SBATCH --job-name fix_seed   ## name that will show up in the queue
#SBATCH -p gpu-preempt
#SBATCH --gpus=4
#SBATCH --mem=50GB  # memory per CPU core
#SBATCH --time=0-15:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --constraint=a100
#SBATCH --constrain=a100|rtx8000|m40
#SBATCH --output=result.txt

source /home/ubuntu/gnn_mini_vs_full/GNNEnv/bin/activate


#vanilla
# python main.py \
#   --dataset pubmed \
#   --dropout 0.7565688403188127 \
#   --lr 0.0001 \
#   --n-partitions 4 \
#   --n-epochs 2000 \
#   --model graphsage \
#   --n-layers 6 \
#   --n-hidden 256 \
#   --log-every 5 \
#   --use-pp \
#   # --fix-seed \
#   # --seed 1344439319 \

# python main.py \
#   --dataset ogbn-arxiv \
#   --dropout 0.3 \
#   --lr 0.003 \
#   --n-partitions 4 \
#   --n-epochs 2000 \
#   --model graphsage \
#   --n-layers 9 \
#   --n-hidden 256 \
#   --log-every 5 \
#   --fix-seed \
#   # --seed 1261325436 \

# python main.py \
#   --dataset reddit \
#   --dropout 0.3 \
#   --lr 0.001 \
#   --n-partitions 4 \
#   --n-epochs 2000 \
#   --model graphsage \
#   --n-layers 7 \
#   --n-hidden 256 \
#   --log-every 5 \
#   --use-pp \
#   # --fix-seed \
#   # --seed 1586505639 \

python main.py \
  --dataset ogbn-products \
  --dropout 0.3 \
  --lr 0.003 \
  --n-partitions 4 \
  --n-epochs 500 \
  --model graphsage \
  --n-layers 5 \
  --n-hidden 256 \
  --log-every 5 \
  --use-pp \
  --fix-seed \
  --seed 837330801 \


# enable pipeline on
# python main.py \
#   --dataset pubmed \
#   --dropout 0.7565688403188127 \
#   --lr 0.0001 \
#   --n-partitions 4 \
#   --n-epochs 2000 \
#   --model graphsage \
#   --n-layers 6 \
#   --n-hidden 256 \
#   --log-every 5 \
#   --use-pp \
#   --enable-pipeline \
#   # --fix-seed \
#   # --seed 1344439319 \

# python main.py \
#   --dataset ogbn-arxiv \
#   --dropout 0.3 \
#   --lr 0.003 \
#   --n-partitions 4 \
#   --n-epochs 2000 \
#   --model graphsage \
#   --n-layers 9 \
#   --n-hidden 256 \
#   --log-every 5 \
#   --enable-pipeline \
#   # --fix-seed \
#   # --seed 1261325436 \

# python main.py \
#   --dataset reddit \
#   --dropout 0.3 \
#   --lr 0.001 \
#   --n-partitions 4 \
#   --n-epochs 2000 \
#   --model graphsage \
#   --n-layers 7 \
#   --n-hidden 256 \
#   --log-every 5 \
#   --enable-pipeline \
#   # --fix-seed \
#   # --seed 1586505639 \

# python main.py \
#   --dataset ogbn-products \
#   --dropout 0.3 \
#   --lr 0.001 \
#   --n-partitions 4 \
#   --n-epochs 2000 \
#   --model graphsage \
#   --n-layers 5 \
#   --n-hidden 256 \
#   --log-every 5 \
#   --use-pp \
#   --enable-pipeline \
#   # --fix-seed \
#   # --seed 22978128 \
