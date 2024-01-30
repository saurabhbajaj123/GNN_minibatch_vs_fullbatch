#!/bin/bash

#SBATCH --job-name quiv-prod   ## name that will show up in the queue
#SBATCH --gpus-per-node=1
#SBATCH --mem=50GB  # memory per CPU core
#SBATCH --time=0-05:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --nodes=1
#SBATCH --partition=gypsum-titanx


nvidia-smi --query-gpu=gpu_name --format=csv,noheader

nvidia-smi topo -m

echo "SLURM_GPUS="$SLURM_GPUS
echo "SLURM_NODELIST = "$SLURM_NODELIST
echo "SLURM_MEM_PER_NODE = "$SLURM_MEM_PER_NODE "MB"
echo "SLURM_JOB_PARTITION = "$SLURM_JOB_PARTITION


cd /work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/pytorch_geometric/torch-quiver
source /work/sbajaj_umass_edu/pygenv1/bin/activate

module load cuda/11.8.0
module load gcc/11.2.0
module load uri/main
module load NCCL/2.12.12-GCCcore-11.3.0-CUDA-11.7.0

QUIVER_ENABLE_CUDA=1 python setup.py install



python3 examples/multi_gpu/ogbn-papers100m/dist_sampling_ogb_products_quiver.py \
  --dataset_subgraph_path preprocess/products_pyg_subgraph.bin \
  --n-epochs 5 \
  --n-gpus 1 \
  --n-layers 5 \
  --n-hidden 128 \
  --batch-size 4096 \
  --eval-batch-size 100000 \
  --weight-decay 0 \
  --fanout 10 \
  --agg mean \
  --log-every 10
