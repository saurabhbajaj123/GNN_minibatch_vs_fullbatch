#!/bin/bash

#SBATCH --job-name orkutdgl   ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=56        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=100G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --partition=gpu-preempt
#SBATCH --constraint=intel8480
#SBATCH --time=00:25:00          # total run time limit (HH:MM:SS)

nvidia-smi --query-gpu=gpu_name --format=csv,noheader
nvidia-smi topo -m
nvidia-smi nvlink -i 0 -s


echo "SLURM_GPUS="$SLURM_GPUS
echo "SLURM_NODELIST = "$SLURM_NODELIST
echo "SLURM_MEM_PER_NODE = "$SLURM_MEM_PER_NODE "MB"
echo "SLURM_JOB_PARTITION = "$SLURM_JOB_PARTITION
echo "Total GPUs ="$(($SLURM_GPUS * $SLURM_NNODES))


source /work/sbajaj_umass_edu/GNNEnv/bin/activate

for n_gpus in 1 2 3 4
do 
  echo "ngpus "$n_gpus
  python main.py \
    --dataset orkut \
    --model gat \
    --sampling NS \
    --dropout 0.5 \
    --lr 0.0005 \
    --n-epochs 5 \
    --n-gpus $n_gpus \
    --n-layers 3 \
    --n-hidden 128 \
    --num-heads 1 \
    --batch-size 1024 \
    --patience 50 \
    --fanout 5 \
    --agg mean \
    --log-every 10 \
    --seed 42 \
    --mode puregpu
done