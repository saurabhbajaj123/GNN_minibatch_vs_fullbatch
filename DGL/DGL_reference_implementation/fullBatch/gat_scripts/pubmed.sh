#!/bin/bash

#SBATCH --job-name pubmed   ## name that will show up in the queue
#SBATCH --gpus=1
#SBATCH --mem=15GB  # memory per CPU core
#SBATCH --time=0-04:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --constraint=m40
#SBATCH --nodes=1

source /work/sbajaj_umass_edu/GNNEnv/bin/activate


python3 gcn_train.py \
  --dataset pubmed \
  --dropout 0.5 \
  --lr 1e-2 \
  --n-epochs 200 \
  --n-layers 2 \
  --n-hidden 16 \
  --log-every 5 \
#   --seed 42 \