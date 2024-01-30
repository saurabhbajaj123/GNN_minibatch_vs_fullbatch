#!/bin/bash

#SBATCH --job-name products-mb   ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --mem=100GB  # memory per CPU core
#SBATCH --time=0-24:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --nodes=1
#SBATCH --partition=gypsum-m40

nvidia-smi --query-gpu=gpu_name --format=csv,noheader

nvidia-smi topo -m


echo "SLURM_GPUS="$SLURM_GPUS
echo "SLURM_NODELIST = "$SLURM_NODELIST
echo "SLURM_MEM_PER_NODE = "$SLURM_MEM_PER_NODE "MB"
echo "SLURM_JOB_PARTITION = "$SLURM_JOB_PARTITION
echo "Total GPUs ="$(($SLURM_GPUS * $SLURM_NNODES))

source /work/sbajaj_umass_edu/GNNEnv/bin/activate

for n_gpu in 1 2 3 4
do
  echo "number of gpus "$n_gpu
  python main.py \
    --dataset ogbn-products \
    --model gat \
    --sampling NS \
    --dropout 0.3 \
    --lr 0.003 \
    --n-epochs 5 \
    --n-gpus $n_gpu \
    --n-layers 3 \
    --n-hidden 128 \
    --num-heads 1 \
    --batch-size 2048 \
    --fanout 10 \
    --patience 50 \
    --agg mean \
    --log-every 10 \
    --seed 42 \
    --mode puregpu
done