#!/bin/bash

#SBATCH --job-name products   ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=100G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --partition=gpu-preempt
#SBATCH --constraint=m40
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)


source /work/sbajaj_umass_edu/GNNEnv/bin/activate


for n_parts in 4
do

  python main.py \
    --dataset ogbn-products \
    --dropout 0.3 \
    --lr 0.007 \
    --n-partitions $n_parts \
    --n-epochs 5 \
    --model graphsage \
    --n-layers 3 \
    --n-hidden 128 \
    --log-every 10 \
    --use-pp \
    --fix-seed \
    --patience 50 \
    --seed 7635837650068751000 \
    --enable-pipeline \
    # --seed 837330801 \
done

for n_parts in 4
do
  python main.py \
    --dataset ogbn-products \
    --dropout 0.3 \
    --lr 0.003 \
    --n-partitions $n_parts \
    --n-epochs 5 \
    --model graphsage \
    --n-layers 3 \
    --n-hidden 128 \
    --log-every 10 \
    --use-pp \
    --fix-seed \
    --patience 50 \
    --seed 7635837650068751000 \
    # --seed 837330801 \
done
