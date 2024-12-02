#!/bin/bash
#SBATCH --job-name=pipe-papers100m     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --cpus-per-task=56        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=250G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:4             # number of gpus per node
#SBATCH --partition=gpu-preempt
#SBATCH --constraint=intel8480
#SBATCH --time=0:10:00          # total run time limit (HH:MM:SS)

source /work/sbajaj_umass_edu/GNNEnv/bin/activate


for n_gpus in 1 2 3 4
  do
    python main.py \
      --dataset ogbn-products \
      --dropout 0.3 \
      --lr 0.003 \
      --n-partitions $n_gpus \
      --n-epochs 1 \
      --model graphsage \
      --n-layers 5 \
      --n-hidden 256 \
      --log-every 10 \
      --fix-seed \
      --patience 50 \
      --seed 7635837650068751000
  done
