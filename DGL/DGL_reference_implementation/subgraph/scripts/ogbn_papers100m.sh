#!/bin/bash

#SBATCH --job-name mbpapers  ## name that will show up in the queue
#SBATCH --mem=200GB  # memory per CPU core
#SBATCH --partition=gypsum-m40
#SBATCH --time=0-04:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --nodes=1
#SBATCH --gpus=4

nvidia-smi --query-gpu=gpu_name --format=csv,noheader
nvidia-smi topo -m
export GLOO_SOCKET_IFNAME=`ip -o -4 route show to default | awk '{print $5}'`


echo "SLURM_NODELIST = "$SLURM_NODELIST
echo "SLURM_MEM_PER_NODE = "$SLURM_MEM_PER_NODE "MB"
echo "SLURM_JOB_PARTITION = "$SLURM_JOB_PARTITION
echo "SLURM_GPUS = "$SLURM_GPUS # 

source /home/ubuntu/gnn_mini_vs_full/GNNEnv/bin/activate

# python papers100m_subgraph.py \
#   --dataset ogbn-arxiv \
#   --n-layers 2 \


python main.py \
  --dataset ogbn-papers100m \
  --dataset-subgraph-path /home/ubuntu/gnn_mini_vs_full/GNN_minibatch_vs_fullbatch/DGL/DGL_reference_implementation/subgraph/ogbn-papers100M_frac_100.0_hops_2_subgraph.bin \
  --model graphsage \
  --sampling NS \
  --dropout 0.3 \
  --lr 0.003 \
  --n-epochs 100 \
  --n-gpus 4 \
  --n-layers 2 \
  --n-hidden 128 \
  --num-heads 2 \
  --batch-size 1024 \
  --patience 50 \
  --fanout 5 \
  --agg mean \
  --log-every 10 \
  --seed 42 \
  # --mode puregpu \
