#!/bin/bash

#SBATCH --job-name ogbn-products   ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --mem=50GB  # memory per CPU core
#SBATCH --time=0-02:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --output=result_products.txt
#SBATCH --constrain=m40
#SBATCH --nodes=1


source /work/sbajaj_umass_edu/GNNEnv/bin/activate

python main.py \
  --dataset ogbn-products \
  --dropout 0.3 \
  --lr 0.001 \
  --n-partitions 4 \
  --n-epochs 1000 \
  --model graphsage \
  --n-layers 5 \
  --n-hidden 256 \
  --log-every 100 \
  --use-pp \
  --fix-seed \
  --patience 50 \
  # --enable-pipeline \
  # --seed 22978128 \

# python main.py \
#   --dataset ogbn-products \
#   --dropout 0.3 \
#   --lr 0.003 \
#   --n-partitions 4 \
#   --n-epochs 500 \
#   --model graphsage \
#   --n-layers 4 \
#   --n-hidden 127 \
#   --log-every 5 \
#   --use-pp \
#   --fix-seed \
#   --seed 837330801 \
#   # --enable-pipeline \


# python main.py \
#   --dataset ogbn-products \
#   --dropout 0.3 \
#   --lr 0.003 \
#   --n-partitions 4 \
#   --n-epochs 500 \
#   --model graphsage \
#   --n-layers 4 \
#   --n-hidden 127 \
#   --log-every 5 \
#   --use-pp \
#   --fix-seed \
#   --enable-pipeline \
#   --seed 837330801 \