#!/bin/bash

#SBATCH --job-name papers-mb   ## name that will show up in the queue
#SBATCH --gpus=1
#SBATCH --mem=500GB  # memory per CPU core
#SBATCH --time=0-12:59:00  ## time for analysis (day-hour:min:sec)
#SBATCH --nodes=1
#SBATCH --partition=gypsum-m40

nvidia-smi --query-gpu=gpu_name --format=csv,noheader

echo "SLURM_GPUS="$SLURM_GPUS
echo "SLURM_NODELIST = "$SLURM_NODELIST
echo "SLURM_MEM_PER_NODE = "$SLURM_MEM_PER_NODE "MB"
echo "SLURM_JOB_PARTITION = "$SLURM_JOB_PARTITION

source /work/sbajaj_umass_edu/GNNEnv/bin/activate


python main.py \
  --dataset ogbn-papers100M \
  --model garphsage \
  --sampling NS \
  --dropout 0.3 \
  --lr 0.01 \
  --n-epochs 300 \
  --n-gpus 4 \
  --n-layers 2 \
  --n-hidden 128 \
  --batch-size 1024 \
  --fanout 4 \
  --patience 50 \
  --agg mean \
  --log-every 5 \
  --seed 42 \
  # --mode puregpu \