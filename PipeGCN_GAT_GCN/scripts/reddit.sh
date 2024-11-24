#!/bin/bash

#SBATCH --job-name reddit   ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --mem=150GB
#SBATCH --time=0-20:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --partition=gypsum-m40
#SBATCH --nodes=1
#SBATCH --exclusive


source /work/sbajaj_umass_edu/GNNEnv/bin/activate

for n_parts in 1 2 3 4 5
do
  python main.py \
      --dataset reddit \
      --dropout 0.3 \
      --lr 0.001 \
      --n-partitions 4 \
      --n-epochs 1000 \
      --model gat \
      --n-layers 2 \
      --n-hidden 1024 \
      --num-heads 2 \
      --log-every 10 \
      --patience 50 \
      --fix-seed \
      --use-pp \
      --seed 42 \
      --enable-pipeline
done

for n_parts in 1 2 3 4 5
do
  python main.py \
      --dataset reddit \
      --dropout 0.3 \
      --lr 0.001 \
      --n-partitions 4 \
      --n-epochs 1000 \
      --model gat \
      --n-layers 2 \
      --n-hidden 1024 \
      --num-heads 2 \
      --log-every 10 \
      --patience 50 \
      --fix-seed \
      --use-pp \
      --seed 42
done


for n_parts in 1 2 3 4 5
do
  python main.py \
      --dataset reddit \
      --dropout 0.3 \
      --lr 0.001 \
      --n-partitions 4 \
      --n-epochs 1000 \
      --model gcn \
      --n-layers 2 \
      --n-hidden 1024 \
      --num-heads 2 \
      --log-every 10 \
      --patience 50 \
      --fix-seed \
      --use-pp \
      --seed 42
done

for n_parts in 1 2 3 4 5
do
  python main.py \
      --dataset reddit \
      --dropout 0.3 \
      --lr 0.001 \
      --n-partitions 4 \
      --n-epochs 1000 \
      --model gcn \
      --n-layers 2 \
      --n-hidden 1024 \
      --num-heads 2 \
      --log-every 10 \
      --patience 50 \
      --fix-seed \
      --use-pp \
      --seed 42 \
      --enable-pipeline
done
