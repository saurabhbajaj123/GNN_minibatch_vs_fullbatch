#!/bin/bash

#SBATCH --job-name pubmed   ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --mem=250GB
#SBATCH --time=0-02:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --partition=gypsum-m40
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12  # cpu-cores per task
#SBATCH --exclusive

source /work/sbajaj_umass_edu/GNNEnv/bin/activate

python main.py \
  --dataset pubmed \
  --dropout 0.7 \
  --lr 0.001 \
  --n-partitions 4 \
  --n-epochs 5 \
  --model graphsage \
  --n-layers 3 \
  --n-hidden 256 \
  --log-every 10 \
  --use-pp \
  --patience 50 \
  --fix-seed \
  --seed 1344439319 \
  --enable-pipeline \

python main.py \
  --dataset pubmed \
  --dropout 0.7 \
  --lr 0.001 \
  --n-partitions 4 \
  --n-epochs 5 \
  --model graphsage \
  --n-layers 3 \
  --n-hidden 256 \
  --log-every 10 \
  --use-pp \
  --patience 50 \
  --fix-seed \
  --seed 1344439319 \
