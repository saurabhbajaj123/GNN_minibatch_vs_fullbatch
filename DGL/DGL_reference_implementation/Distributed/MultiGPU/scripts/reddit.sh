#!/bin/bash

#SBATCH --job-name reddit  ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --mem=100GB  # memory per CPU core
#SBATCH --time=0-24:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --partition=gypsum-m40
#SBATCH --nodes=1



source /home/ubuntu/gnn_mini_vs_full/GNNEnv/bin/activate
for n_parts in 1 2 3 4
do
  echo $n_parts
  python main.py \
    --dataset reddit \
    --model graphsage  \
    --sampling NS \
    --dropout 0.3 \
    --lr 0.001 \
    --n-epochs 5 \
    --n-gpus $n_parts \
    --n-layers 5 \
    --n-hidden 512 \
    --num-heads 2 \
    --batch-size 512 \
    --fanout 4 \
    --agg mean \
    --log-every 10 \
    --patience 50 \
    --seed 42 \
    --mode puregpu
done