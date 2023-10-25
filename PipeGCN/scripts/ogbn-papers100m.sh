#!/bin/bash

#SBATCH --job-name 100m  ## name that will show up in the queue
#SBATCH --partition=gpu-preempt
#SBATCH --constraint=a100
#SBATCH --gpus=8
#SBATCH --mem=250GB  # memory per CPU core
#SBATCH --time=0-00:01:00  ## time for analysis (day-hour:min:sec)



source /work/sbajaj_umass_edu/GNNEnv/bin/activate

python main.py \
  --dataset ogbn-papers100m \
  --dropout 0.3 \
  --lr 0.003 \
  --n-partitions 16 \
  --n-epochs 20 \
  --model graphsage \
  --n-layers 1 \
  --n-hidden 16 \
  --log-every 10 \
  --enable-pipeline \
  --use-pp

  # --partition-method parmetis \