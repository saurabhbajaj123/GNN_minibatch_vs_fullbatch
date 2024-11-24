#!/bin/bash

#SBATCH --job-name bnsreddit   ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=100G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --partition=gypsum-m40
#SBATCH --time=20:00:00          # total run time limit (HH:MM:SS)
#SBATCH --exclusive

export GLOO_SOCKET_IFNAME=`ip -o -4 route show to default | awk '{print $5}'`

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "MASTER_PORT="$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

echo "NUM GPUS PER NODE="$SLURM_GPUS 

source /work/sbajaj_umass_edu/GNNEnv/bin/activate

for n_parts in 1 2 3 4 5
do
  echo $n_parts
  python main.py \
    --dataset reddit \
    --dropout 0.5 \
    --lr 0.01 \
    --n-partitions 4 \
    --n-epochs 1000 \
    --model graphsage \
    --sampling-rate .1 \
    --n-layers 4 \
    --n-hidden 1024 \
    --heads 4 \
    --log-every 10 \
    --use-pp
done


for n_parts in 1 2 3 4 5
do
  echo $n_parts
  python main.py \
    --dataset reddit \
    --dropout 0.5 \
    --lr 0.01 \
    --n-partitions 4 \
    --n-epochs 1000 \
    --model gat \
    --sampling-rate .1 \
    --n-layers 2 \
    --n-hidden 1024 \
    --heads 2 \
    --log-every 10 \
    --use-pp
done

for n_parts in 1 2 3 4 5
do
  echo $n_parts
  python main.py \
    --dataset reddit \
    --dropout 0.5 \
    --lr 0.01 \
    --n-partitions 4 \
    --n-epochs 1000 \
    --model gcn \
    --sampling-rate .1 \
    --n-layers 2 \
    --n-hidden 1024 \
    --heads 4 \
    --log-every 10 \
    --use-pp
done