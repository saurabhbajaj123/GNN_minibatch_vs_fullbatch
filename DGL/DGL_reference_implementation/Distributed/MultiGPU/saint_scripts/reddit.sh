#!/bin/bash

#SBATCH --job-name reddit  ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --mem=100GB  # memory per CPU core
#SBATCH --time=0-24:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --partition=gypsum-m40
#SBATCH --nodes=1


source /work/sbajaj_umass_edu/GNNEnv/bin/activate

python saint_main.py \
  --dataset reddit \
  --model graphsage \
  --sampling NS \
  --dropout 0.5 \
  --lr 0.001 \
  --n-epochs 200 \
  --n-gpus 4 \
  --n-layers 3 \
  --n-hidden 256 \
  --batch-size 1024 \
  --budget_node_edge 4000 \
  --num_iters 1000 \
  --budget_rw_0 256 \
  --budget_rw_1 256 \
  --agg mean \
  --log-every 20 \
#   --seed 10245829 \
