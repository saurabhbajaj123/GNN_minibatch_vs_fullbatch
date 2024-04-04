#!/bin/bash

#SBATCH --job-name arxiv   ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --mem=250GB
#SBATCH --time=0-00:20:00  ## time for analysis (day-hour:min:sec)
#SBATCH --partition=gpu-preempt
#SBATCH --constraint=intel8480
#SBATCH --exclude=superpod-gpu[001-005]
#SBATCH --nodes=1
#SBATCH --cpus-per-task=112  # cpu-cores per task


source /work/sbajaj_umass_edu/GNNEnv/bin/activate

for n_parts in 1 2 3 4
do
  python main.py \
    --dataset ogbn-arxiv \
    --dropout 0.5 \
    --lr 0.005 \
    --n-partitions $n_parts \
    --n-epochs 5 \
    --model gat \
    --num-heads 6 \
    --n-layers 3 \
    --n-hidden 1024 \
    --log-every 10 \
    --patience 50 \
    --use-pp \
    --seed 42 \
    --enable-pipeline
done


for n_parts in 1 2 3 4
do
  python main.py \
    --dataset ogbn-arxiv \
    --dropout 0.5 \
    --lr 0.005 \
    --n-partitions $n_parts \
    --n-epochs 5 \
    --model gat \
    --num-heads 6 \
    --n-layers 3 \
    --n-hidden 1024 \
    --log-every 10 \
    --patience 50 \
    --use-pp \
    --seed 42
done