#!/bin/bash

#SBATCH --job-name papers-mb   ## name that will show up in the queue
#SBATCH --gpus=1
#SBATCH --mem=250GB  # memory per CPU core
#SBATCH --time=0-24:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --constraint=avx512
#SBATCH --partition=gpu-preempt
#SBATCH --nodes=1


source /work/sbajaj_umass_edu/GNNEnv/bin/activate


python main.py \
  --dataset ogbn-papers100M \
  --model garphsage \
  --sampling NS \
  --dropout 0.3 \
  --lr 0.001 \
  --n-epochs 100 \
  --n-gpus 1 \
  --n-layers 1 \
  --n-hidden 16 \
  --num-heads 2 \
  --batch-size 256 \
  --fanout 4 \
  --patience 50 \
  --agg mean \
  --log-every 10 \
  --seed 42 \
  # --mode puregpu \