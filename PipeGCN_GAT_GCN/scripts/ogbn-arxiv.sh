#!/bin/bash

#SBATCH --job-name arxiv   ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --mem=50GB
#SBATCH --time=0-20:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --partition=gypsum-m40
#SBATCH --nodes=1
#SBATCH --exclusive


source /work/sbajaj_umass_edu/GNNEnv/bin/activate

for n_parts in 1 2 3 4 5
do
  python main.py \
    --dataset ogbn-arxiv \
    --dropout 0.5 \
    --lr 0.005 \
    --n-partitions 4 \
    --n-epochs 1000 \
    --model gat \
    --num-heads 2 \
    --n-layers 3 \
    --n-hidden 1024 \
    --log-every 10 \
    --patience 50 \
    --use-pp \
    --seed 42 \
    --enable-pipeline
done


for n_parts in 1 2 3 4 5
do
  python main.py \
    --dataset ogbn-arxiv \
    --dropout 0.5 \
    --lr 0.005 \
    --n-partitions $n_parts \
    --n-epochs 1000 \
    --model gat \
    --num-heads 2 \
    --n-layers 3 \
    --n-hidden 1024 \
    --log-every 10 \
    --patience 50 \
    --use-pp \
    --seed 42
done