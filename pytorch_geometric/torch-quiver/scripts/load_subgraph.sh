#!/bin/bash

#SBATCH --job-name quiv-subg   ## name that will show up in the queue
#SBATCH --gpus=1
#SBATCH --mem=20GB  # memory per CPU core
#SBATCH --time=0-24:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --nodes=1
#SBATCH --partition=gypsum-m40

cd /work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/pytorch_geometric/torch-quiver
source /work/sbajaj_umass_edu/pygenv1/bin/activate

module load cuda/11.8.0
module load gcc/11.2.0
module load uri/main
module load NCCL/2.12.12-GCCcore-11.3.0-CUDA-11.7.0

QUIVER_ENABLE_CUDA=1 python setup.py install


python preprocess/load_subgraph.py