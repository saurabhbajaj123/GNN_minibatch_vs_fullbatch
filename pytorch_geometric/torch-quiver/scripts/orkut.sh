#!/bin/bash

#SBATCH --job-name quiv-ork   ## name that will show up in the queue
#SBATCH --gpus-per-node=4
#SBATCH --mem=200GB  # memory per CPU core
#SBATCH --time=0-24:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --nodes=1
#SBATCH --partition=gypsum-1080ti


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


for n_parts in 1 2 3 4
do
  echo "using "$n_parts" GPUS"
  python3 examples/multi_gpu/pyg/orkut/dist_sampling_orkut_quiver.py \
    --model graphsage \
    --n-epochs 5 \
    --n-gpus $n_parts \
    --n-layers 2 \
    --n-hidden 128 \
    --batch-size 1024 \
    --eval-batch-size 100000 \
    --weight-decay 0 \
    --fanout 10 \
    --heads 1 \
    --agg mean \
    --log-every 10
done

# python3 examples/pyg/ogbn_products_sage_quiver.py