#!/bin/bash

#SBATCH --job-name 100m  ## name that will show up in the queue
#SBATCH -c 32
#SBATCH --partition=cpu
#SBATCH --mem=700GB  # memory per CPU core
#SBATCH --time=0-24:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --nodes=2

echo "NUM NODES="$SLURM_JOB_NUM_NODES

export GLOO_SOCKET_IFNAME=`ip -o -4 route show to default | awk '{print $5}'`

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "MASTER_PORT="$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

source /work/sbajaj_umass_edu/GNNEnv/bin/activate

python main.py \
  --dataset ogbn-papers100m \
  --dropout 0.3 \
  --lr 0.003 \
  --n-partitions 64 \
  --n-epochs 30 \
  --model graphsage \
  --n-layers 2 \
  --n-hidden 64 \
  --log-every 5 \
  --enable-pipeline \
  --use-pp \
  --parts-per-node 32 \
  --skip-partition \
  --fix-seed \
  # --partition-method parmetis \