# #!/bin/bash

# #SBATCH --job-name arx-quiver   ## name that will show up in the queue
# #SBATCH --gpus=4
# #SBATCH --mem=40GB  # memory per CPU core
# #SBATCH --time=0-01:00:00  ## time for analysis (day-hour:min:sec)
# #SBATCH --nodes=1
# #SBATCH --partition=gypsum-m40

# nvidia-smi --query-gpu=gpu_name --format=csv,noheader

# nvidia-smi topo -m

# echo "SLURM_GPUS="$SLURM_GPUS
# echo "SLURM_NODELIST = "$SLURM_NODELIST
# echo "SLURM_MEM_PER_NODE = "$SLURM_MEM_PER_NODE "MB"
# echo "SLURM_JOB_PARTITION = "$SLURM_JOB_PARTITION


# cd /work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/pytorch_geometric/torch-quiver
# source /home/ubuntu/gnn_mini_vs_full/pygenv1/bin/activate

# module load cuda/11.8.0
# module load gcc/11.2.0
# module load uri/main
# module load NCCL/2.12.12-GCCcore-11.3.0-CUDA-11.7.0

# QUIVER_ENABLE_CUDA=1 python setup.py install


for n_gpus in 4
do
  echo $n_gpus 
  python3 examples/multi_gpu/pyg/ogb-arxiv/dist_sampling_ogb_arxiv_quiver.py \
    --dataset ogbn-arxiv \
    --model graphsage \
    --sampling NS \
    --dropout 0.3 \
    --lr 0.001 \
    --n-epochs 5 \
    --batch-size 1024 \
    --n-gpus $n_gpus \
    --n-layers 2 \
    --n-hidden 512 \
    --weight-decay 0 \
    --fanout 5 \
    --agg mean \
    --log-every 10 \
    --seed 12345
done