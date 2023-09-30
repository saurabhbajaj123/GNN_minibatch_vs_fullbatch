#!/bin/bash

#SBATCH --job-name ogbn-products   ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --mem=50GB  # memory per CPU core
#SBATCH --time=0-08:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --constrain=m40
#SBATCH --nodes=1


source /work/sbajaj_umass_edu/GNNEnv/bin/activate


python main.py \
  --dataset ogbn-products \
  --model graphsage \
  --sampling NS \
  --dropout 0.3 \
  --lr 0.001 \
  --n-epochs 500 \
  --n-gpus 4 \
  --n-layers 6 \
  --n-hidden 128 \
  --batch-size 1024 \
  --fanout 4 \
  --patience 100 \
  --agg mean \
  --log-every 10 \
  --seed 42 \
  --mode puregpu \
#   --seed 10245829 \
  # --seed \