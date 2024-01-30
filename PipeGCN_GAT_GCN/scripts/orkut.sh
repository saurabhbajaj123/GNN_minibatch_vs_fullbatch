#!/bin/bash
#SBATCH --job-name=ddp-orkut     # create a short name for your job
#SBATCH --nodes=3                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --mem=80G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gpus-per-node=4             # number of gpus per node
#SBATCH --partition=gpu-preempt
#SBATCH --constraint=m40
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
# export NUM_PARTITIONS=$(( $SLURM_GPUS * $SLURM_NNODES ))
# echo "NUM_PARTITIONS="$NUM_PARTITIONS

source /work/sbajaj_umass_edu/GNNEnv/bin/activate

echo "orkut scalability multi node =" $WORLD_SIZE "nodes"

srun python main.py \
  --dataset orkut \
  --dropout 0.3 \
  --lr 0.002 \
  --n-partitions 12 \
  --n-epochs 5 \
  --model graphsage \
  --num-heads 1 \
  --n-layers 2 \
  --n-hidden 128 \
  --log-every 10 \
  --patience 100 \
  --fix-seed \
  --seed 42 \
  --use-pp \
  --master-addr $MASTER_ADDR \
  --port $MASTER_PORT \
  --parts-per-node 4 \
  --skip-partition \
  --enable-pipeline \
