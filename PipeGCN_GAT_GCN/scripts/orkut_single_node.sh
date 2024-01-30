#!/bin/bash

#SBATCH --job-name products   ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --mem=100GB  # memory per CPU core
#SBATCH --time=0-24:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --partition=gypsum-m40
#SBATCH --nodes=1

source /work/sbajaj_umass_edu/GNNEnv/bin/activate

for n_parts in 2 3 4
do
  echo "scalability orkut gat single node"
  echo "num parts = "$n_parts
  python main.py \
    --dataset orkut \
    --dropout 0.3 \
    --lr 0.002 \
    --n-partitions $n_parts \
    --n-epochs 5 \
    --model gat \
    --num-heads 1 \
    --n-layers 2 \
    --n-hidden 128 \
    --log-every 10 \
    --patience 100 \
    --fix-seed \
    --seed 42 \
    --enable-pipeline
done