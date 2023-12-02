#!/bin/bash

#SBATCH --job-name arxiv  ## name that will show up in the queue
#SBATCH --mem=40GB  # memory per CPU core
#SBATCH --partition=gypsum-m40
#SBATCH --time=0-02:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --nodes=1
#SBATCH --gpus=2

nvidia-smi --query-gpu=gpu_name --format=csv,noheader
nvidia-smi topo -m
export GLOO_SOCKET_IFNAME=`ip -o -4 route show to default | awk '{print $5}'`


echo "SLURM_NODELIST = "$SLURM_NODELIST
echo "SLURM_MEM_PER_NODE = "$SLURM_MEM_PER_NODE "MB"
echo "SLURM_JOB_PARTITION = "$SLURM_JOB_PARTITION
echo "SLURM_GPUS = "$SLURM_GPUS # 

source /work/sbajaj_umass_edu/GNNEnv/bin/activate

# python papers100m_subgraph.py \
#   --dataset ogbn-arxiv \
#   --n-layers 2 \


python main.py \
  --dataset ogbn-arxiv \
  --model graphsage \
  --sampling NS \
  --dropout 0.3 \
  --lr 0.001 \
  --n-epochs 10 \
  --n-gpus $SLURM_GPUS \
  --n-layers 5 \
  --n-hidden 128 \
  --num-heads 2 \
  --batch-size 1024 \
  --patience 50 \
  --fanout 4 \
  --agg mean \
  --log-every 5 \
  --seed 42 \
  # --mode puregpu \

