#!/bin/bash

#SBATCH --job-name pap-quiver   ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --mem=250GB
#SBATCH --time=0-02:20:00  ## time for analysis (day-hour:min:sec)
#SBATCH --partition=gpu-preempt
#SBATCH --constraint=intel8480
#SBATCH --exclude=superpod-gpu[004-005]
#SBATCH --nodes=1
#SBATCH --cpus-per-task=112  # cpu-cores per task
#SBATCH --exclusive

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

python3 examples/multi_gpu/ogbn-papers100m/dist_sampling_ogb_papers100m_quiver.py \
  --dataset_subgraph_path preprocess/papers100M_pyg_subgraph.bin \
  --n-epochs 5 \
  --n-gpus 4 \
  --n-layers 2 \
  --n-hidden 128 \
  --batch-size 1024 \
  --eval-batch-size 100000 \
  --weight-decay 0 \
  --fanout 5 \
  --agg mean \
  --log-every 10 \
