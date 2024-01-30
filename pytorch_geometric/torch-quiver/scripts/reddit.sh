#!/bin/bash

#SBATCH --job-name quiv-red   ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --mem=100GB  # memory per CPU core
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

for n_parts in 1 2 3 4
do
  echo $n_parts
    python3 examples/multi_gpu/pyg/reddit/dist_sampling_ogb_reddit_quiver.py \
    --n-epochs 5 \
    --n-gpus $n_parts \
    --fanout 15 \
    --n_hidden 1024 \
    --n_layers 2 \
    --heads 1 \
    --batch_size 1024 \
    --log_every 10
done
# python3 examples/pyg/reddit_quiver.py