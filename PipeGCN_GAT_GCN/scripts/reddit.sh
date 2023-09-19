#!/bin/bash

#SBATCH --job-name ogbn-products   ## name that will show up in the queue
#SBATCH -p gpu-preempt
#SBATCH --gpus=4
#SBATCH --mem=50GB  # memory per CPU core
#SBATCH --time=0-00:30:00  ## time for analysis (day-hour:min:sec)
#SBATCH --output=result.txt


source /work/sbajaj_umass_edu/GNNEnv/bin/activate

python main.py \
  --dataset reddit \
  --dropout 0.3 \
  --lr 0.001 \
  --n-partitions 4 \
  --n-epochs 1000 \
  --model gat \
  --n-layers 2 \
  --n-hidden 128 \
  --num-heads 3 \
  --log-every 5 \
  --fix-seed \
  --seed 1586505639 \
  --use-pp \
  # --enable-pipeline \
  # --inductive \