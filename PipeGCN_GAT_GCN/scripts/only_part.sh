#!/bin/bash

#SBATCH --job-name 100m  ## name that will show up in the queue
#SBATCH --partition=cpu
#SBATCH --mem=200GB  # memory per CPU core
#SBATCH --time=0-24:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --nodes=1


source /work/sbajaj_umass_edu/GNNEnv/bin/activate

srun python main.py \
  --dataset orkut \
  --dropout 0.3 \
  --lr 0.002 \
  --n-partitions 12 \
  --n-epochs 5 \
  --model gat \
  --num-heads 3 \
  --n-layers 3 \
  --n-hidden 128 \
  --log-every 10 \
  --patience 100 \
  --fix-seed \
  --seed 42 \
  # --enable-pipeline \
