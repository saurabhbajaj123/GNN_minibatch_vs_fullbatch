#!/bin/bash

#SBATCH --job-name products   ## name that will show up in the queue
#SBATCH --gpus=1
#SBATCH --mem=50GB  # memory per CPU core
#SBATCH --time=0-24:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --constraint=m40

source /work/sbajaj_umass_edu/GNNEnv/bin/activate

python3 gcn_train.py \
  --dataset ogbn-products \
  --dropout 0.3 \
  --lr 0.003 \
  --n-epochs 2000 \
  --n-layers 2 \
  --n-hidden 128 \
  --num-heads 2 \
  --patience 100 \
  --log-every 10 \
  --seed 42 \