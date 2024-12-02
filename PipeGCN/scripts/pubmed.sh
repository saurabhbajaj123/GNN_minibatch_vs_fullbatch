#!/bin/bash

#SBATCH --job-name pubmed   ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --mem=50GB
#SBATCH --time=0-00:10:00  ## time for analysis (day-hour:min:sec)
#SBATCH --partition=gypsum-titanx
#SBATCH --nodes=1

source /work/sbajaj_umass_edu/GNNEnv/bin/activate

# for n_layers in 2 3 4 5 6
# do
#   for n_hidden in 64 128 256 512 1024
#   do 
# for n_gpus in 1 2 3 4
# do
python main.py \
  --dataset pubmed \
  --dropout 0.7 \
  --lr 0.001 \
  --n-partitions 4 \
  --n-epochs 1 \
  --model graphsage \
  --n-layers 3 \
  --n-hidden 256 \
  --log-every 10 \
  --patience 50 \
  --fix-seed \
  --seed 1344439319 \
  # done
  # done