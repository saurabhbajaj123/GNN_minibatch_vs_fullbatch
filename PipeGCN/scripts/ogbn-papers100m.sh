#!/bin/bash
#SBATCH --job-name=ddp-papers100m     # create a short name for your job
#SBATCH --nodes=8                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --mem=120G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:4             # number of gpus per node
#SBATCH --partition=gypsum-titanx
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)

export GLOO_SOCKET_IFNAME=`ip -o -4 route show to default | awk '{print $5}'`

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "MASTER_PORT="$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

source /home/ubuntu/gnn_mini_vs_full/GNNEnv/bin/activate

srun python main.py \
  --dataset ogbn-papers100m \
  --dropout 0.3 \
  --lr 0.003 \
  --n-partitions 64 \
  --n-epochs 100 \
  --model graphsage \
  --n-layers 1 \
  --n-hidden 16 \
  --log-every 5 \
  --enable-pipeline \
  --use-pp \
  --patience 50 \
  --parts-per-node 4 \
  --master-addr $MASTER_ADDR \
  --port $MASTER_PORT \
  --fix-seed \
  --skip-partition \
  # --partition-method parmetis \