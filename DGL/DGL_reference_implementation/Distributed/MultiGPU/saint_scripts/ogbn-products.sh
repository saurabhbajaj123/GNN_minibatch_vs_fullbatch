#!/bin/bash

#SBATCH --job-name products   ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --mem=100GB  # memory per CPU core
#SBATCH --time=0-24:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --constraint=m40
#SBATCH --nodes=1


source /work/sbajaj_umass_edu/GNNEnv/bin/activate

python saint_main.py \
  --dataset ogbn-products \
  --model graphsage \
  --sampling saint \
  --dropout 0.3 \
  --lr 0.001 \
  --n-epochs 200 \
  --n-gpus 4 \
  --n-layers 3 \
  --n-hidden 256 \
  --batch-size 2048 \
  --budget_node_edge 10000 \
  --budget_rw_0 256 \
  --budget_rw_1 16 \
  --mode_saint node \
  --num-iters 1000 \
  --log-every 20 \
#   --seed 10245829 \

  