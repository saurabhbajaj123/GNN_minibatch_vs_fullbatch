# python3 -m venv pygenv1
# srun -p gpu-preempt -t 01:59:00 --gpus=4 --mem=80GB --partition=gypsum-m40 --nodes=1 --pty /bin/bash
# srun -p gpu-preempt -t 01:59:00 --gpus=2 --mem=20GB --constraint=avx512 --nodes=1 --pty /bin/bash
cd /work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/pytorch_geometric/torch-quiver
source /work/sbajaj_umass_edu/pygenv1/bin/activate

# pip3 uninstall torch torchvision torchaudio
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip install torch_geometric torch-sparse
# pip3 install torch_scatter

# pip install ogb
# pip install wandb
# pip install torchmetrics
module load cuda/11.8.0
module load gcc/11.2.0
module load uri/main
module load NCCL/2.12.12-GCCcore-11.3.0-CUDA-11.7.0

QUIVER_ENABLE_CUDA=1 python setup.py install
