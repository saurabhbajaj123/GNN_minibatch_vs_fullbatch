#!/bin/bash

#SBATCH --job-name orkpip   ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=56        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=100G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --partition=gpu-preempt
#SBATCH --constraint=intel8480
#SBATCH --time=00:50:00          # total run time limit (HH:MM:SS)

source /work/sbajaj_umass_edu/GNNEnv/bin/activate

for n_parts in 1 2 3 4
do
  echo "scalability orkut graphsage single node"
  echo "num parts = "$n_parts
  python main.py \
    --dataset orkut \
    --dropout 0.3 \
    --lr 0.002 \
    --n-partitions $n_parts \
    --n-epochs 5 \
    --model graphsage \
    --num-heads 2 \
    --n-layers 3 \
    --n-hidden 128 \
    --log-every 10 \
    --patience 100 \
    --fix-seed \
    --seed 42 \
    --enable-pipeline
done


for n_parts in 1 2 3 4
do
  echo "scalability orkut graphsage single node"
  echo "num parts = "$n_parts
  python main.py \
    --dataset orkut \
    --dropout 0.3 \
    --lr 0.002 \
    --n-partitions $n_parts \
    --n-epochs 5 \
    --model graphsage \
    --num-heads 2 \
    --n-layers 3 \
    --n-hidden 128 \
    --log-every 10 \
    --patience 100 \
    --fix-seed \
    --seed 42
done