#!/bin/bash

#SBATCH --job-name arxiv  ## name that will show up in the queue
#SBATCH --partition=cpu
#SBATCH -c 64
#SBATCH --mem=250GB  # memory per CPU core
#SBATCH --time=0-24:00:00  ## time for analysis (day-hour:min:sec)



source /work/sbajaj_umass_edu/GNNEnv/bin/activate


# python sage_train.py \
#   --dataset ogbn-products \
#   --n-epochs 500 \
#   --n-layers 5 \
#   --n-hidden 512 \
#   --lr 0.001 \
#   --dropout 0.5 \
#   --seed 42 \
#   --num_partitions 4000 \
#   --agg mean \
#   --batch_size 256 \
#   --log_every 5 \

# python sage_train.py \
#   --dataset ogbn-arxiv \
#   --n-epochs 1000 \
#   --n-layers 6 \
#   --n-hidden 512 \
#   --lr 0.001 \
#   --dropout 0.5 \
#   --seed 42 \
#   --num_partitions 4000 \
#   --agg mean \
#   --batch_size 256 \
#   --log_every 5 \

# python sage_train.py \
#   --dataset reddit \
#   --n-epochs 200 \
#   --n-layers 4 \
#   --n-hidden 1024 \
#   --device_id 0 \
#   --lr 0.001 \
#   --dropout 0.5 \
#   --seed 42 \
#   --num_partitions 4000 \
#   --agg mean \
#   --batch_size 256 \
#   --log_every 5 \

python sage_train.py \
  --dataset pubmed \
  --n-epochs 200 \
  --n-layers 3 \
  --n-hidden 1024 \
  --device_id 0 \
  --lr 0.004 \
  --dropout 0.5 \
  --seed 42 \
  --num_partitions 4000 \
  --agg mean \
  --batch_size 256 \
  --log_every 5 \
