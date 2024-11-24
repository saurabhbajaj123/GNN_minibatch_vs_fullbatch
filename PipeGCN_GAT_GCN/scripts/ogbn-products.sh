#!/bin/bash

#SBATCH --job-name products   ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --mem=250GB
#SBATCH --time=0-20:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --partition=gypsum-m40
#SBATCH --nodes=1

source /work/sbajaj_umass_edu/GNNEnv/bin/activate


for n_parts in 1 2 3 4
do
  python main.py \
    --dataset ogbn-products \
    --dropout 0.3 \
    --lr 0.001 \
    --n-partitions $n_parts \
    --n-epochs 5 \
    --model gat \
    --num-heads 3 \
    --n-layers 3 \
    --n-hidden 128 \
    --log-every 10 \
    --patience 100 \
    --fix-seed \
    --seed 42 \
    --enable-pipeline
done

for n_parts in 1 2 3 4
do
  python main.py \
    --dataset ogbn-products \
    --dropout 0.3 \
    --lr 0.001 \
    --n-partitions $n_parts \
    --n-epochs 5 \
    --model gat \
    --num-heads 3 \
    --n-layers 3 \
    --n-hidden 128 \
    --log-every 10 \
    --patience 100 \
    --fix-seed \
    --seed 42
done