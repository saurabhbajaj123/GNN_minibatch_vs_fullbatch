#!/bin/bash

#SBATCH --job-name pubmed   ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --mem=50GB
#SBATCH --time=0-00:10:00  ## time for analysis (day-hour:min:sec)
#SBATCH --partition=gypsum-1080ti
#SBATCH --nodes=1


source /work/sbajaj_umass_edu/GNNEnv/bin/activate


# for n_layers in 2 3 4 5 6
# do
#   for n_hidden in 64 128 256 512 1024
#   do 
#     for batch_size in 256 512 1024 2048 4096
#     do
#       for fanout in 5 10 15 20 25
#       do  
for n_gpus in 4
  do
    python main.py \
      --dataset pubmed \
      --model graphsage \
      --sampling NS \
      --dropout 0.3 \
      --lr 0.001 \
      --n-epochs 100 \
      --n-gpus $n_gpus \
      --n-layers 3 \
      --n-hidden 256 \
      --num-heads 4 \
      --batch-size 1024 \
      --patience 50 \
      --fanout 10 \
      --agg mean \
      --log-every 10 \
      --seed 42 \
      --mode puregpu
  done


# for n_gpus in 1
#   do
#     python main.py \
#       --dataset pubmed \
#       --model gcn \
#       --sampling NS \
#       --dropout 0.3 \
#       --lr 0.001 \
#       --n-epochs 300 \
#       --n-gpus 1 \
#       --n-layers 6 \
#       --n-hidden 64 \
#       --num-heads 4 \
#       --batch-size 1024 \
#       --patience 50 \
#       --fanout 10 \
#       --agg mean \
#       --log-every 10 \
#       --seed 42 \
#       --mode puregpu
#   done

  #     done
  #   done
  # done


# python main.py \
#   --dataset pubmed \
#   --model graphsage \
#   --sampling NS \
#   --dropout 0.3 \
#   --lr 0.003 \
#   --n-epochs 1000 \
#   --n-gpus 4 \
#   --n-layers 3 \
#   --n-hidden 187 \
#   --batch-size 1024 \
#   --patience 50 \
#   --fanout 4 \
#   --agg mean \
#   --log-every 5 \
#   --mode puregpu \
#   --seed 42 \
#   # --seed \