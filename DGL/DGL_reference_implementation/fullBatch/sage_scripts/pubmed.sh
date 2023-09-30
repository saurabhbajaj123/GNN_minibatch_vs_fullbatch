#!/bin/bash

#SBATCH --job-name test   ## name that will show up in the queue
#SBATCH -p gpu-preempt
#SBATCH --gpus=1
#SBATCH --mem=50GB  # memory per CPU core
#SBATCH --time=0-00:01:00  ## time for analysis (day-hour:min:sec)
#SBATCH --constraint=a100
#SBATCH --output=result.txt

source /work/sbajaj_umass_edu/GNNEnv/bin/activate

python3 sage_train.py \
  --dataset pubmed \
  --dropout 0.55 \
  --lr 1e-4 \
  --n-epochs 1000 \
  --n-layers 10 \
  --n-hidden 512 \
  --log-every 5 \
  --agg gcn \
  # --device_id 1 \
#   --seed 42 \