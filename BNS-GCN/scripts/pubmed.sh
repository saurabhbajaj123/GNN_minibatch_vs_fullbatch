#!/bin/bash

#SBATCH --job-name bnspubmed   ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --mem=50GB
#SBATCH --time=0-20:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --partition=gypsum-m40
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12  # cpu-cores per task
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

for dummy in 1 2 3 4 5; do
  python main.py \
    --dataset pubmed \
    --dropout 0.3 \
    --lr 0.001 \
    --n-epochs 1000 \
    --n-partitions 4 \
    --model graphsage \
    --sampling-rate 0.1 \
    --n-layers 4 \
    --n-hidden 64 \
    --log-every 10 \
    --use-pp \
    # --seed \
done

for dummy in 1 2 3 4 5; do
python main.py \
  --dataset pubmed \
  --dropout 0.3 \
  --lr 0.001 \
  --n-epochs 1000 \
  --n-partitions 4 \
  --model gat \
  --heads 1 \
  --sampling-rate 0.1 \
  --n-layers 3 \
  --n-hidden 1024 \
  --log-every 10 \
  --use-pp \
  # --seed \
done

for dummy in 1 2 3 4 5; do
python main.py \
  --dataset pubmed \
  --dropout 0.3 \
  --lr 0.001 \
  --n-epochs 5 \
  --n-partitions 4 \
  --model gcn \
  --sampling-rate 0.1 \
  --n-layers 2 \
  --n-hidden 512 \
  --log-every 10 \
  --use-pp \
  # --seed \
done

