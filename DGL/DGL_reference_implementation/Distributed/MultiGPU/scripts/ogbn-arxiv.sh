#!/bin/bash

#SBATCH --job-name ogbn-arxiv   ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --mem=100GB  # memory per CPU core
#SBATCH --time=0-24:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --partition=gypsum-m40
#SBATCH --nodes=1


source /home/ubuntu/gnn_mini_vs_full/GNNEnv/bin/activate
for n_parts in 2 3 4
do
  echo $n_parts
  python main.py \
    --dataset ogbn-arxiv \
    --model graphsage \
    --sampling NS \
    --dropout 0.3 \
    --lr 0.007 \
    --n-epochs 5 \
    --n-gpus $n_parts \
    --n-layers 6 \
    --n-hidden 1024 \
    --num-heads 2 \
    --batch-size 1024 \
    --patience 200 \
    --fanout 5 \
    --agg mean \
    --log-every 10 \
    --seed 42 \
    --mode puregpu
done