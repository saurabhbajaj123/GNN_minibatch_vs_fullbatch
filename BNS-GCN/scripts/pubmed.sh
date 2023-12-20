#!/bin/bash

#SBATCH --job-name bnspubmed   ## name that will show up in the queue
#SBATCH --gpus=1
#SBATCH --mem=20GB  # memory per CPU core
#SBATCH --time=0-24:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --partition=gpu-preempt
#SBATCH --constraint=m40
#SBATCH --nodes=1

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

for layers in 2 3 4 5
do
  for hidden in 64 128 256
  do
    python main.py \
      --dataset pubmed \
      --dropout 0.3 \
      --lr 0.0007 \
      --n-epochs 100 \
      --n-partitions 1 \
      --model graphsage \
      --sampling-rate .1 \
      --n-layers $layers \
      --n-hidden $hidden \
      --log-every 5 \
      --use-pp \
      # --seed \
  done
done