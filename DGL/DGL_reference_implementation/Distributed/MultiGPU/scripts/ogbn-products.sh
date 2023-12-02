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


# source /work/sbajaj_umass_edu/GNNEnv/bin/activate

# python main.py \
#   --dataset ogbn-products \
#   --model garphsage \
#   --sampling NS \
#   --dropout 0.3 \
#   --lr 0.001 \
#   --n-epochs 100 \
#   --n-gpus 1 \
#   --n-layers 2 \
#   --n-hidden 128 \
#   --num-heads 2 \
#   --batch-size 1024 \
#   --fanout 4 \
#   --patience 50 \
#   --agg mean \
#   --log-every 5 \
#   --seed 42 \
#   # --mode puregpu \
