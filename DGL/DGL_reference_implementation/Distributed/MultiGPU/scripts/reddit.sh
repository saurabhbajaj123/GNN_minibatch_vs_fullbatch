#!/bin/bash

#SBATCH --job-name reddit  ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=100G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --partition=gypsum-m40
#SBATCH --time=08:00:00          # total run time limit (HH:MM:SS)
#SBATCH --exclusive

nvidia-smi --query-gpu=gpu_name --format=csv,noheader

nvidia-smi topo -m


echo "SLURM_GPUS="$SLURM_GPUS
echo "SLURM_NODELIST = "$SLURM_NODELIST
echo "SLURM_MEM_PER_NODE = "$SLURM_MEM_PER_NODE "MB"
echo "SLURM_JOB_PARTITION = "$SLURM_JOB_PARTITION
echo "Total GPUs ="$(($SLURM_GPUS * $SLURM_NNODES))

# echo "cpu cores 56"
source /work/sbajaj_umass_edu/GNNEnv/bin/activate


# for n_layers in 4
# do
#   for n_hidden in 1024
#   do 
#     for batch_size in 256 512
#     do
#       for fanout in 25
#       do  

for n_gpus in 3
  do
    python main.py \
      --dataset reddit \
      --model graphsage  \
      --sampling NS \
      --dropout 0.3 \
      --lr 0.001 \
      --n-epochs 10 \
      --n-gpus $n_gpus \
      --n-layers 4 \
      --n-hidden 1024 \
      --num-heads 2 \
      --batch-size 1024 \
      --fanout 5 \
      --agg mean \
      --log-every 2 \
      --patience 50 \
      --seed 42 
      # --mode puregpu
    done

# for n_gpus in 1
#   do
#     python main.py \
#       --dataset reddit \
#       --model gcn  \
#       --sampling NS \
#       --dropout 0.3 \
#       --lr 0.001 \
#       --n-epochs 300 \
#       --n-gpus 4 \
#       --n-layers 2 \
#       --n-hidden 512 \
#       --num-heads 1 \
#       --batch-size 1024 \
#       --fanout 5 \
#       --agg mean \
#       --log-every 15 \
#       --patience 50 \
#       --seed 42 \
#       --mode puregpu
#     done


#     done
#   done
# done
