#!/bin/bash

#SBATCH --job-name pubmed   ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --mem=12GB  # memory per CPU core
#SBATCH --time=0-04:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --partition=gypsum-m40
#SBATCH --nodes=1


source /work/sbajaj_umass_edu/GNNEnv/bin/activate

python main.py \
  --dataset pubmed \
  --model graphsage \
  --sampling NS \
  --dropout 0.3 \
  --lr 0.001 \
  --n-epochs 10 \
  --n-gpus 1 \
  --n-layers 7 \
  --n-hidden 1024 \
  --num-heads 4 \
  --batch-size 1024 \
  --patience 50 \
  --fanout 10 \
  --agg mean \
  --log-every 5 \
  --seed 42 \
  --mode puregpu \
  # --seed \


# python main.py \
#   --dataset pubmed \
#   --model graphsage \
#   --sampling NS \
#   --dropout 0.3 \
#   --lr 0.003 \
#   --n-epochs 1000 \
#   --n-gpus 4 \
#   --n-layers 3 \
#   --n-hidden 187 \
#   --batch-size 1024 \
#   --patience 50 \
#   --fanout 4 \
#   --agg mean \
#   --log-every 5 \
#   --mode puregpu \
#   --seed 42 \
#   # --seed \