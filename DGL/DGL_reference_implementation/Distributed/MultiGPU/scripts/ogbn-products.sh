#!/bin/bash

#SBATCH --job-name products-mb   ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=100G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --partition=gpu-preempt
#SBATCH --constraint=m40
#SBATCH --time=00:59:00          # total run time limit (HH:MM:SS)

nvidia-smi --query-gpu=gpu_name --format=csv,noheader

nvidia-smi topo -m


echo "SLURM_GPUS="$SLURM_GPUS
echo "SLURM_NODELIST = "$SLURM_NODELIST
echo "SLURM_MEM_PER_NODE = "$SLURM_MEM_PER_NODE "MB"
echo "SLURM_JOB_PARTITION = "$SLURM_JOB_PARTITION
echo "Total GPUs ="$(($SLURM_GPUS * $SLURM_NNODES))

source /work/sbajaj_umass_edu/GNNEnv/bin/activate

for n_gpu in 4
do
  echo "number of gpus "$n_gpu
  python main.py \
    --dataset ogbn-products \
    --model graphsage \
    --sampling NS \
    --dropout 0.3 \
    --lr 0.003 \
    --n-epochs 100 \
    --n-gpus $n_gpu \
    --n-layers 3  \
    --n-hidden 128 \
    --num-heads 1 \
    --batch-size 1024 \
    --fanout 20 \
    --patience 50 \
    --agg mean \
    --log-every 10 \
    --seed 42 \
    --mode puregpu
done