#!/bin/bash

#SBATCH --job-name 100m  ## name that will show up in the queue
#SBATCH -c 32
#SBATCH --partition=cpu-long
#SBATCH --mem=700GB  # memory per CPU core
#SBATCH --time=3-00:00:00  ## time for analysis (day-hour:min:sec)



source /work/sbajaj_umass_edu/GNNEnv/bin/activate

python main.py \
  --dataset ogbn-papers100m \
  --dropout 0.3 \
  --lr 0.003 \
  --n-partitions 32 \
  --n-epochs 100 \
  --model graphsage \
  --n-layers 1 \
  --n-hidden 16 \
  --log-every 10 \
  --enable-pipeline \
  --use-pp

  # --partition-method parmetis \