#!/bin/bash
#SBATCH --job-name=ddp-prod     # create a short name for your job
#SBATCH --nodes=2                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --mem=64G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gpus-per-node=4             # number of gpus per node
#SBATCH --time=02:00:00          # total run time limit (HH:MM:SS)
#SBATCH --partition=gypsum-m40

export GLOO_SOCKET_IFNAME=`ip -o -4 route show to default | awk '{print $5}'`

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "MASTER_PORT="$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

export NUM_PARTS=$($SLURM_GPUS*$SLURM_NNODES) 

# echo "NUM_PARTS="$NUM_PARTS
echo "SLURM_STEP_GPUS="$SLURM_STEP_GPUS
echo "SLURM_JOB_GPUS="$SLURM_JOB_GPUS
echo "SLURM_GPUS="$SLURM_GPUS

source /work/sbajaj_umass_edu/GNNEnv/bin/activate

srun python main.py \
  --dataset ogbn-products \
  --dropout 0.3 \
  --lr 0.003 \
  --n-partitions 8 \
  --n-epochs 1000 \
  --model graphsage \
  --n-layers 3 \
  --n-hidden 128 \
  --log-every 5 \
  --patience 50 \
  --fix-seed \
  --master-addr $MASTER_ADDR \
  --port $MASTER_PORT \
  --seed 1261325436 \
  --parts-per-node 4 \
  --backend gloo \
  --skip-partition \
  --enable-pipeline \