#!/bin/bash

#SBATCH --job-name pubmed   ## name that will show up in the queue
#SBATCH --gpus=2
#SBATCH --mem=25GB
#SBATCH --time=0-00:30:00  ## time for analysis (day-hour:min:sec)
#SBATCH --partition=gypsum-m40
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2  # cpu-cores per task


source /work/sbajaj_umass_edu/GNNEnv/bin/activate

python cluster_main.py \
  --dataset pubmed \
  --model graphsage \
  --sampling cluster \
  --dropout 0.3 \
  --lr 0.003 \
  --n-epochs 50 \
  --n-gpus 1 \
  --n-layers 3 \
  --n-hidden 256 \
  --batch-size 1024 \
  --num-partitions 2000 \
  --agg mean \
  --log-every 5 \
  --seed 6238418958544123000 \
  --mode puregpu \
  # --seed \