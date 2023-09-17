#!/bin/bash

#SBATCH --job-name test   ## name that will show up in the queue
#SBATCH -p gpu-preempt
#SBATCH --gpus=1
#SBATCH --mem=20GB  # memory per CPU core
#SBATCH --time=0-05:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --constraint=a100
#SBATCH --output=result.txt

source /work/sbajaj_umass_edu/GNNEnv/bin/activate

python3 sage_train.py \
  --dataset ogbn-products \
  --dropout 0.5 \
  --lr 0.003 \
  --n-epochs 2000 \
  --n-layers 4 \
  --n-hidden 128 \
  --log-every 5 \
  # --seed 42 \