#!/bin/bash

#SBATCH --job-name products-mb   ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --mem=100G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --partition=gpu-preempt
#SBATCH --constraint=2080ti
#SBATCH --time=00:59:00          # total run time limit (HH:MM:SS)
#SBATCH --exclusive


nvidia-smi --query-gpu=gpu_name --format=csv,noheader

nvidia-smi topo -m

echo "SLURM_GPUS="$SLURM_GPUS
echo "SLURM_NODELIST = "$SLURM_NODELIST
echo "SLURM_MEM_PER_NODE = "$SLURM_MEM_PER_NODE "MB"
echo "SLURM_JOB_PARTITION = "$SLURM_JOB_PARTITION
echo "Total GPUs ="$(($SLURM_GPUS * $SLURM_NNODES))

source /work/sbajaj_umass_edu/GNNEnv/bin/activate


for n_gpus in 4
do 
  echo "n_gpus "$n_gpus
  python main.py \
    --dataset pubmed \
    --model gat \
    --sampling NS \
    --dropout 0.3 \
    --lr 0.0005 \
    --n-epochs 1 \
    --n-gpus $n_gpus \
    --n-layers 3 \
    --n-hidden 256 \
    --num-heads 2 \
    --batch-size 1024 \
    --fanout 10 \
    --agg mean \
    --log-every 10 \
    --seed 42 
    # --mode puregpu
done



for n_gpus in 4
do 
  echo "n_gpus "$n_gpus
  python main.py \
    --dataset ogbn-arxiv \
    --model gat \
    --sampling NS \
    --dropout 0.3 \
    --lr 0.0005 \
    --n-epochs 1 \
    --n-gpus $n_gpus \
    --n-layers 4 \
    --n-hidden 256 \
    --num-heads 2 \
    --batch-size 1024 \
    --fanout 20 \
    --agg mean \
    --log-every 10 \
    --seed 42 
    # --mode puregpu
done

for n_gpus in 4
do 
  echo "n_gpus "$n_gpus
  python main.py \
    --dataset reddit \
    --model gat \
    --sampling NS \
    --dropout 0.3 \
    --lr 0.0005 \
    --n-epochs 1 \
    --n-gpus $n_gpus \
    --n-layers 2 \
    --n-hidden 512 \
    --num-heads 2 \
    --batch-size 1024 \
    --fanout 5 \
    --agg mean \
    --log-every 10 \
    --seed 42 
    # --mode puregpu
done

for n_gpus in 4
do 
  echo "n_gpus "$n_gpus
  python main.py \
    --dataset ogbn-products \
    --model gat \
    --sampling NS \
    --dropout 0.3 \
    --lr 0.0005 \
    --n-epochs 1 \
    --n-gpus 4 \
    --n-layers 3 \
    --n-hidden 128 \
    --num-heads 2 \
    --batch-size 1024 \
    --fanout 5 \
    --agg mean \
    --log-every 10 \
    --seed 42 
    # --mode puregpu
done



for n_gpus  in 4; do

  python main.py \
    --dataset ogbn-papers100m \
    --dataset-subgraph-path /work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/DGL/DGL_reference_implementation/subgraph/ogbn-papers100M_frac_100.0_hops_2_subgraph.bin \
    --model gat \
    --sampling NS \
    --dropout 0.3 \
    --lr 0.001 \
    --n-epochs 1 \
    --n-gpus $n_gpus \
    --n-layers 2 \
    --n-hidden 128 \
    --num-heads 1 \
    --batch-size 1024 \
    --fanout 5 \
    --agg mean \
    --log-every 5 \
    --seed 42 
    # --mode puregpu
  done

python main.py \
  --dataset ogbn-papers100M \
  --dataset-subgraph-path /work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/DGL/DGL_reference_implementation/subgraph/ogbn-papers100M_frac_100.0_hops_2_subgraph.bin \
  --model gat \
  --sampling NS \
  --dropout 0.3 \
  --lr 0.001 \
  --n-epochs 300 \
  --n-gpus $n_gpus \
  --n-layers 2 \
  --n-hidden 128 \
  --num-heads 1 \
  --batch-size 1024 \
  --fanout 5 \
  --agg mean \
  --log-every 5 \
  --seed 42 
  # --mode puregpu