#!/bin/bash

#SBATCH --job-name reddit   ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --mem=250GB
#SBATCH --time=0-00:20:00  ## time for analysis (day-hour:min:sec)
#SBATCH --partition=gpu-preempt
#SBATCH --constraint=intel8480
#SBATCH --exclude=superpod-gpu[001-005]
#SBATCH --nodes=1
#SBATCH --cpus-per-task=112  # cpu-cores per task
#SBATCH --reservation=hgxbenchmark
#SBATCH -A pi_mserafini_umass_edu

source /work/sbajaj_umass_edu/GNNEnv/bin/activate

for n_parts in 1 2 3 4
do
  echo "num parts" $n_parts
  python main.py \
      --dataset reddit \
      --dropout 0.3 \
      --lr 0.001 \
      --n-partitions $n_parts \
      --n-epochs 5 \
      --model gat \
      --n-layers 2 \
      --n-hidden 1024 \
      --num-heads 4 \
      --log-every 10 \
      --patience 50 \
      --fix-seed \
      --use-pp \
      --seed 42 \
      --enable-pipeline
done

for n_parts in 1 2 3 4
do
  echo "num parts" $n_parts
  python main.py \
      --dataset reddit \
      --dropout 0.3 \
      --lr 0.001 \
      --n-partitions $n_parts \
      --n-epochs 5 \
      --model gat \
      --n-layers 2 \
      --n-hidden 1024 \
      --num-heads 4\
      --log-every 10 \
      --patience 50 \
      --fix-seed \
      --use-pp \
      --seed 42
done