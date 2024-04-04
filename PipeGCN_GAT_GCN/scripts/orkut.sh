#!/bin/bash
#SBATCH --job-name=ddp-orkut     # create a short name for your job
#SBATCH --gpus=4
#SBATCH --mem=250GB
#SBATCH --time=0-00:20:00  ## time for analysis (day-hour:min:sec)
#SBATCH --partition=gpu-preempt
#SBATCH --constraint=intel8480
#SBATCH --exclude=superpod-gpu[001-005]
#SBATCH --nodes=1
#SBATCH --cpus-per-task=112  # cpu-cores per task
#SBATCH --reservation=hgxbenchmark
#SBATCH -A pi_mserafini_umass_edu
#SBATCH --exclusive

export GLOO_SOCKET_IFNAME=`ip -o -4 route show to default | awk '{print $5}'`

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
# export NUM_PARTITIONS=$(( $SLURM_GPUS * $SLURM_NNODES ))
# echo "NUM_PARTITIONS="$NUM_PARTITIONS

source /work/sbajaj_umass_edu/GNNEnv/bin/activate

echo "orkut scalability multi node =" $WORLD_SIZE "nodes"

srun python main.py \
  --dataset orkut \
  --dropout 0.3 \
  --lr 0.002 \
  --n-partitions 4 \
  --n-epochs 5 \
  --model graphsage \
  --n-layers 3 \
  --n-hidden 512 \
  --log-every 10 \
  --patience 100 \
  --fix-seed \
  --seed 42 \
  --use-pp \
  --master-addr $MASTER_ADDR \
  --port $MASTER_PORT \
  --parts-per-node 4 \
  --skip-partition \
  # --enable-pipeline \
