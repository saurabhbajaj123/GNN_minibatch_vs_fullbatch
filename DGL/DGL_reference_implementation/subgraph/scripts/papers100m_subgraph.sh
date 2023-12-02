#!/bin/bash

#SBATCH --job-name arxiv  ## name that will show up in the queue
#SBATCH --mem=40GB  # memory per CPU core
#SBATCH --partition=cpu
#SBATCH --time=0-02:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --nodes=1



echo "SLURM_GPUS="$SLURM_GPUS
echo "SLURM_NODELIST = "$SLURM_NODELIST
echo "SLURM_MEM_PER_NODE = "$SLURM_MEM_PER_NODE "MB"
echo "SLURM_JOB_PARTITION = "$SLURM_JOB_PARTITION

source /work/sbajaj_umass_edu/GNNEnv/bin/activate

python papers100m_subgraph.py \
  --dataset ogbn-arxiv \
  --n-layers 2 \


# python main.py \
#   --dataset ogbn-arxiv \
#   --model graphsage \
#   --sampling NS \
#   --dropout 0.3 \
#   --lr 0.001 \
#   --n-epochs 1000 \
#   --n-gpus 4 \
#   --n-layers 5 \
#   --n-hidden 128 \
#   --num-heads 2 \
#   --batch-size 1024 \
#   --patience 50 \
#   --fanout 4 \
#   --agg mean \
#   --log-every 5 \
#   --seed 42 \
#   --mode puregpu \

