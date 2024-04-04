#!/bin/bash
#SBATCH --job-name=bns-orkut     # create a short name for your job
#SBATCH --nodes=3                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --cpus-per-task=112        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=250G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:4             # number of gpus per node
#SBATCH --partition=gpu-preempt
#SBATCH --constraint=intel8480
#SBATCH --exclude=superpod-gpu[001-005]
#SBATCH --time=01:00:00          # total run time limit (HH:MM:SS)

#SBATCH -A pi_mserafini_umass_edu


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
  --sampling-rate 0.1 \
  --n-layers 3 \
  --n-hidden 512 \
  --log-every 10 \
  --fix-seed \
  --seed 42 \
  --use-pp \
  --master-addr $MASTER_ADDR \
  --port $MASTER_PORT \
  --parts-per-node 4 \
  # --skip-partition \
