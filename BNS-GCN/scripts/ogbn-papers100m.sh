#!/bin/bash
#SBATCH --job-name=bnspapers100m     # create a short name for your job
#SBATCH --nodes=3                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --mem=250G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gpus-per-node=4             # number of gpus per node
#SBATCH --partition=gypsum-m40
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)

export GLOO_SOCKET_IFNAME=`ip -o -4 route show to default | awk '{print $5}'`

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "MASTER_PORT="$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

echo "NUM GPUS PER NODE="$SLURM_GPUS 
echo "Nodes = "$SLURM_JOB_NODELIST

source /work/sbajaj_umass_edu/GNNEnv/bin/activate

srun python main.py \
  --dataset ogbn-papers100m \
  --dataset-subgraph-path /work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/DGL/DGL_reference_implementation/subgraph/ogbn-papers100M_frac_100.0_hops_2_subgraph.bin \
  --dropout 0.3 \
  --lr 0.01 \
  --n-partitions 12 \
  --n-epochs 1000 \
  --model graphsage \
  --sampling-rate 0.1 \
  --n-layers 2 \
  --n-hidden 128 \
  --log-every 10 \
  --fix-seed \
  --use-pp \
  --master-addr $MASTER_ADDR \
  --port $MASTER_PORT \
  --parts-per-node 4 \
