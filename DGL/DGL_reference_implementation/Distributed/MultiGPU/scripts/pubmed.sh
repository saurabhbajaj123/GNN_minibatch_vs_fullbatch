#!/bin/bash

#SBATCH --job-name pubmed   ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --mem=50GB
#SBATCH --time=0-00:10:00  ## time for analysis (day-hour:min:sec)
#SBATCH --partition=gypsum-1080ti
#SBATCH --nodes=1


source /work/sbajaj_umass_edu/GNNEnv/bin/activate

for n_gpus in 4
  do
    python main.py \
      --dataset pubmed \
      --model graphsage \
      --sampling NS \
      --dropout 0.3 \
      --lr 0.001 \
      --n-epochs 50 \
      --n-gpus $n_gpus \
      --n-layers 3 \
      --n-hidden 256 \
      --num-heads 4 \
      --batch-size 1024 \
      --patience 50 \
      --fanout 10 \
      --agg mean \
      --log-every 10 \
      --seed 42 \
      --mode puregpu
  done
