#!/bin/bash

#SBATCH --job-name pubmed   ## name that will show up in the queue
#SBATCH --gpus=1
#SBATCH --mem=15GB  # memory per CPU core
#SBATCH --time=0-24:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --partition=gypsum-titanx
#SBATCH --nodes=1

source /work/sbajaj_umass_edu/GNNEnv/bin/activate

for n_parts in 1
do
  python main.py \
    --dataset pubmed \
    --dropout 0.5 \
    --lr 0.001 \
    --n-partitions $n_parts \
    --n-epochs 1 \
    --model gat \
    --n-layers 3 \
    --n-hidden 256 \
    --num-heads 2 \
    --log-every 5 \
    --patience 50 \
    --use-pp \
    --seed 42 \
    --enable-pipeline
done


# for n_parts in 1 2 3 4 5
# do
#   python main.py \
#     --dataset pubmed \
#     --dropout 0.5 \
#     --lr 0.001 \
#     --n-partitions 4 \
#     --n-epochs 500 \
#     --model gat \
#     --n-layers 3 \
#     --n-hidden 1024 \
#     --num-heads 1 \
#     --log-every 5 \
#     --patience 50 \
#     --use-pp \
#     --seed 42
# done


# for n_parts in 1 2 3 4 5
# do
#   python main.py \
#     --dataset pubmed \
#     --dropout 0.5 \
#     --lr 0.001 \
#     --n-partitions 4 \
#     --n-epochs 500 \
#     --model gcn \
#     --n-layers 2 \
#     --n-hidden 512 \
#     --num-heads 1 \
#     --log-every 5 \
#     --patience 50 \
#     --use-pp \
#     --seed 42 \
#     --enable-pipeline
# done


# for n_parts in 1 2 3 4 5
# do
#   python main.py \
#     --dataset pubmed \
#     --dropout 0.5 \
#     --lr 0.001 \
#     --n-partitions 4 \
#     --n-epochs 500 \
#     --model gcn \
#     --n-layers 2 \
#     --n-hidden 512 \
#     --num-heads 1 \
#     --log-every 5 \
#     --patience 50 \
#     --use-pp \
#     --seed 42
# done