#!/bin/bash

#SBATCH --job-name arxiv  ## name that will show up in the queue
#SBATCH --partition=cpu
#SBATCH -c 64
#SBATCH --mem=250GB  # memory per CPU core
#SBATCH --time=0-24:00:00  ## time for analysis (day-hour:min:sec)


source /work/sbajaj_umass_edu/GNNEnv/bin/activate


# python cluster_main.py \
#   --dataset ogbn-products \
#   --model graphsage \
#   --sampling cluster \
#   --dropout 0.3 \
#   --lr 0.002 \
#   --n-epochs 500 \
#   --n-gpus 4 \
#   --n-layers 4 \
#   --n-hidden 728 \
#   --batch-size 512 \
#   --num-partitions 4000 \
#   --agg mean \
#   --log-every 5 \
#   --seed 3485963027166655500 \
# #   --mode puregpu \
#   # --seed \


# python cluster_main.py \
#   --dataset reddit \
#   --model graphsage \
#   --sampling cluster \
#   --dropout 0.3 \
#   --lr 0.003 \
#   --n-epochs 1000 \
#   --n-gpus 4 \
#   --n-layers 4 \
#   --n-hidden 728 \
#   --batch-size 1024 \
#   --num-partitions 4000 \
#   --agg mean \
#   --log-every 5 \
#   --seed 42 \
# #   --mode puregpu \
#   # --seed \

# python cluster_main.py \
#   --dataset ogbn-arxiv \
#   --model graphsage \
#   --sampling cluster \
#   --dropout 0.3 \
#   --lr 0.001 \
#   --n-epochs 1000 \
#   --n-gpus 4 \
#   --n-layers 6 \
#   --n-hidden 512 \
#   --batch-size 1024 \
#   --num-partitions 4000 \
#   --agg mean \
#   --log-every 5 \
# #   --seed 10245829 \
# #   --mode puregpu \
#   # --seed \


python cluster_main.py \
  --dataset pubmed \
  --model graphsage \
  --sampling cluster \
  --dropout 0.3 \
  --lr 0.003 \
  --n-epochs 500 \
  --n-gpus 4 \
  --n-layers 10 \
  --n-hidden 256 \
  --batch-size 1024 \
  --num-partitions 2000 \
  --agg mean \
  --log-every 5 \
#   --seed 10245829 \
#   --mode puregpu \
  # --seed \