#!/bin/bash

#SBATCH --job-name reddit   ## name that will show up in the queue
#SBATCH --gpus=1
#SBATCH --mem=40GB  # memory per CPU core
#SBATCH --time=0-24:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --partition=gypsum-m40
#SBATCH --nodes=1


source /work/sbajaj_umass_edu/GNNEnv/bin/activate

python3 sage_train.py \
  --dataset reddit \
  --dropout 0.3 \
  --lr 0.003 \
  --n-epochs 1000 \
  --n-layers  3 \
  --n-hidden 256 \
  --log-every 5 \
  --seed 42 \