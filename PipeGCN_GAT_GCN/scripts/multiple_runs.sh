#!/bin/bash

#SBATCH --job-name pubmed   ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --mem=250GB  # memory per CPU core
#SBATCH --time=0-00:30:00  ## time for analysis (day-hour:min:sec)
#SBATCH --nodes=1
#SBATCH -A pi_mserafini_umass_edu
#SBATCH -p superpod-a100


source /work/sbajaj_umass_edu/GNNEnv/bin/activate

# for n_parts in 4
# do
#   python main.py \
#     --dataset pubmed \
#     --dropout 0.5 \
#     --lr 0.001 \
#     --n-partitions $n_parts \
#     --n-epochs 1 \
#     --model gat \
#     --n-layers 3 \
#     --n-hidden 256 \
#     --num-heads 2 \
#     --log-every 5 \
#     --patience 50 \
#     --use-pp \
#     --seed 42 \
#     --enable-pipeline
# done

# for n_parts in 4
# do
#   python main.py \
#       --dataset ogbn-arxiv \
#       --dropout 0.3 \
#       --lr 0.001 \
#       --n-partitions 4 \
#       --n-epochs 1000 \
#       --model gat \
#       --n-layers 4 \
#       --n-hidden 256 \
#       --num-heads 2 \
#       --log-every 10 \
#       --patience 50 \
#       --fix-seed \
#       --use-pp \
#       --seed 42 \
#       --enable-pipeline
# done

# for n_parts in 4
# do
#   python main.py \
#     --dataset reddit \
#     --dropout 0.3 \
#     --lr 0.001 \
#     --n-partitions $n_parts \
#     --n-epochs 5 \
#     --model gat \
#     --num-heads 2 \
#     --n-layers 2 \
#     --n-hidden 512 \
#     --log-every 10 \
#     --patience 100 \
#     --fix-seed \
#     --seed 42 \
#     --enable-pipeline
# done

# for n_parts in 4
# do
#   python main.py \
#     --dataset ogbn-products \
#     --dropout 0.3 \
#     --lr 0.001 \
#     --n-partitions $n_parts \
#     --n-epochs 5 \
#     --model gat \
#     --num-heads 2 \
#     --n-layers 3 \
#     --n-hidden 128 \
#     --log-every 10 \
#     --patience 100 \
#     --fix-seed \
#     --seed 42 \
#     --enable-pipeline
# done



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

for n_gpus in 4; do
    srun python main.py \
      --dataset ogbn-papers100m \
      --dataset-subgraph-path /work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/DGL/DGL_reference_implementation/subgraph/ogbn-papers100M_frac_100.0_hops_2_subgraph.bin \
      --dropout 0.3 \
      --lr 0.01 \
      --n-partitions $n_gpus \
      --n-epochs 1 \
      --model gat \
      --n-layers 2 \
      --n-hidden 128 \
      --num-heads 1 \
      --log-every 10 \
      --patience 50 \
      --fix-seed \
      --master-addr $MASTER_ADDR \
      --port $MASTER_PORT \
      --parts-per-node 4
      # --enable-pipeline \
      # --skip-partition \
    done
  # done
