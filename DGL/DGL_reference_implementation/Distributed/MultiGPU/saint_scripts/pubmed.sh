#!/bin/bash

#SBATCH --job-name pubmed   ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --mem=12GB  # memory per CPU core
#SBATCH --time=0-04:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --constraint=m40
#SBATCH --nodes=1


source /work/sbajaj_umass_edu/GNNEnv/bin/activate


python saint_main.py \
  --dataset pubmed \
  --model graphsage \
  --sampling saint \
  --dropout 0.3 \
  --lr 0.003 \
  --n-epochs 1000 \
  --n-gpus 4 \
  --n-layers 3 \
  --n-hidden 128 \
  --batch-size 256 \
  --budget_node_edge 3000 \
  --num-iters 1000 \
  --budget_rw_0 256 \
  --budget_rw_1 16 \
  --mode_saint node \
  --log-every 5 \
  --patience 100 \
  # --mode puregpu \
#   --seed 10245829 \
