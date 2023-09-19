#!/bin/bash

#SBATCH --job-name ogbn-products   ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --mem=50GB  # memory per CPU core
#SBATCH --time=0-08:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --output=result_products.txt
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
  --n-layers 4 \
  --n-hidden 127 \
  --batch-size 1024 \
  --fanout 4 \
  --patience 50 \
  --agg mean \
  --log-every 5 \
  --seed 42 \
  --mode puregpu \
#   --seed 10245829 \
  # --seed \