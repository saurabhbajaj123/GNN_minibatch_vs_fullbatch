#!/bin/bash

#SBATCH --job-name reddit  ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --mem=50GB  # memory per CPU core
#SBATCH --time=0-24:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --constraint=m40
#SBATCH --nodes=1


source /work/sbajaj_umass_edu/GNNEnv/bin/activate

python main.py \
  --dataset reddit \
  --model graphsage \
  --sampling NS \
  --dropout 0.3 \
  --lr 0.001 \
  --n-epochs 10 \
  --n-gpus 4 \
  --n-layers 12 \
  --n-hidden 256 \
  --batch-size 256 \
  --fanout 4 \
  --agg mean \
  --log-every 5 \
  --seed 42 \
  # --mode puregpu \
  # --seed \


# python main.py \
#   --dataset reddit \
#   --model graphsage \
#   --sampling NS \
#   --dropout 0.3 \
#   --lr 0.001 \
#   --n-epochs 1000 \
#   --n-gpus 4 \
#   --n-layers 4 \
#   --n-hidden 312 \
#   --batch-size 1024 \
#   --fanout 4 \
#   --agg mean \
#   --log-every 5 \
#   --mode puregpu \
#   --seed 42 \
#   # --seed \