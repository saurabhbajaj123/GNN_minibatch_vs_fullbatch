#!/bin/bash

#SBATCH --job-name papers  ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=56        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=250G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --partition=gpu
#SBATCH --time=08:00:00          # total run time limit (HH:MM:SS)
#SBATCH --reservation=hojae-a100

nvidia-smi --query-gpu=gpu_name --format=csv,noheader

nvidia-smi topo -m


export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "MASTER_PORT="$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

echo "NUM GPUS PER NODE="$SLURM_GPUS 

echo "SLURM_GPUS="$SLURM_GPUS
echo "SLURM_NODELIST = "$SLURM_NODELIST
echo "SLURM_MEM_PER_NODE = "$SLURM_MEM_PER_NODE "MB"
echo "SLURM_JOB_PARTITION = "$SLURM_JOB_PARTITION
echo "Total GPUs ="$(($SLURM_GPUS * $SLURM_NNODES))

source /work/sbajaj_umass_edu/GNNEnv/bin/activate


# for n_layers in 2 3 4 5 6
# do
#   for n_hidden in 64 128 256 512 1024
#   do 
#     for batch_size in 512 1024 2048 4096
#     do
#       for fanout in 5 10 15 20 25
#       do  

for n_gpus in 1 2 3 4
do
  python main.py \
    --dataset ogbn-papers100M \
    --dataset-subgraph-path /work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/DGL/DGL_reference_implementation/subgraph/ogbn-papers100M_frac_100.0_hops_2_subgraph.bin \
    --model graphsage \
    --sampling NS \
    --dropout 0.3 \
    --lr 0.003 \
    --n-epochs 200 \
    --n-gpus $n_gpus \
    --n-layers 2 \
    --n-hidden 128 \
    --num-heads 2 \
    --batch-size 1024 \
    --fanout 5 \
    --patience 200 \
    --agg mean \
    --log-every 10 \
    --seed 42 \
    --mode puregpu
  done
  #     done
  #   done
  # done