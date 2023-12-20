#!/bin/bash

#SBATCH --job-name 100m  ## name that will show up in the queue
#SBATCH -c 32
#SBATCH --partition=cpu
#SBATCH --mem=700GB  # memory per CPU core
#SBATCH --time=0-24:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --nodes=1

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
  --dataset-subgraph-path /work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/DGL/DGL_reference_implementation/subgraph/ogbn-papers100M_frac_100.0_hops_2_subgraph.bin \
  --dropout 0.3 \
  --lr 0.003 \
  --n-partitions 1 \
  --n-epochs 10 \
  --model graphsage \
  --n-layers 2 \
  --n-hidden 16 \
  --log-every 5 \
  # --use-pp \
  # --fix-seed \
  # --enable-pipeline \
  # --parts-per-node 1 \
  # --skip-partition \
  # --partition-method parmetis \