#!/bin/bash

#SBATCH --job-name products   ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --mem=100GB  # memory per CPU core
#SBATCH --time=0-24:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --partition=gypsum-m40
#SBATCH --nodes=1


source /work/sbajaj_umass_edu/GNNEnv/bin/activate

python main.py \
  --dataset ogbn-products \
  --dropout 0.3 \
  --lr 0.007 \
  --n-partitions 4 \
  --n-epochs 1000 \
  --model graphsage \
  --n-layers 5 \
  --n-hidden 256 \
  --log-every 10 \
  --use-pp \
  --fix-seed \
  --patience 50 \
  --seed 7635837650068751000 \
  # --seed 837330801 \



# python main.py \
#   --dataset ogbn-products \
#   --dropout 0.3 \
#   --lr 0.003 \
#   --n-partitions 4 \
#   --n-epochs 1000 \
#   --model graphsage \
#   --n-layers 5 \
#   --n-hidden 256 \
#   --log-every 10 \
#   --use-pp \
#   --fix-seed \
#   --patience 50 \
#   --seed 7635837650068751000 \
#   --enable-pipeline \
#   # --seed 837330801 \

