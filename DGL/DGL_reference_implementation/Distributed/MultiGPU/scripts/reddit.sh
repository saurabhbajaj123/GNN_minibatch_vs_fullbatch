#!/bin/bash

#SBATCH --job-name reddit  ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --mem=100GB  # memory per CPU core
#SBATCH --time=0-24:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --partition=gypsum-m40
#SBATCH --nodes=1



source /work/sbajaj_umass_edu/GNNEnv/bin/activate

# python main.py \
#   --dataset reddit \
#   --model graphsage  \
#   --sampling NS \
#   --dropout 0.3 \
#   --lr 0.001 \
#   --n-epochs 10 \
#   --n-gpus 4 \
#   --n-layers 5 \
#   --n-hidden 1024 \
#   --num-heads 2 \
#   --batch-size 4096 \
#   --fanout 4 \
#   --agg mean \
#   --log-every 5 \
#   --patience 50 \
#   --seed 42 \
#   --mode puregpu \

# python main.py \
#   --dataset reddit \
#   --model graphsage  \
#   --sampling NS \
#   --dropout 0.3 \
#   --lr 0.001 \
#   --n-epochs 10 \
#   --n-gpus 4 \
#   --n-layers 5 \
#   --n-hidden 1024 \
#   --num-heads 2 \
#   --batch-size 4096 \
#   --fanout 4 \
#   --agg mean \
#   --log-every 5 \
#   --patience 50 \
#   --seed 42 \
#   # --mode puregpu \

# python main.py \
#   --dataset reddit \
#   --model gat  \
#   --sampling NS \
#   --dropout 0.3 \
#   --lr 0.001 \
#   --n-epochs 10 \
#   --n-gpus 4 \
#   --n-layers 4 \
#   --n-hidden 512 \
#   --num-heads 2 \
#   --batch-size 4096 \
#   --fanout 4 \
#   --agg mean \
#   --log-every 5 \
#   --patience 50 \
#   --seed 42 \
#   --mode puregpu \

# python main.py \
#   --dataset reddit \
#   --model gat  \
#   --sampling NS \
#   --dropout 0.3 \
#   --lr 0.001 \
#   --n-epochs 10 \
#   --n-gpus 4 \
#   --n-layers 4 \
#   --n-hidden 512 \
#   --num-heads 2 \
#   --batch-size 4096 \
#   --fanout 4 \
#   --agg mean \
#   --log-every 5 \
#   --patience 50 \
#   --seed 42 \
#   # --mode puregpu \

# python main.py \
#   --dataset reddit \
#   --model gcn  \
#   --sampling NS \
#   --dropout 0.3 \
#   --lr 0.001 \
#   --n-epochs 10 \
#   --n-gpus 4 \
#   --n-layers 6 \
#   --n-hidden 1024 \
#   --num-heads 2 \
#   --batch-size 4096 \
#   --fanout 4 \
#   --agg mean \
#   --log-every 5 \
#   --patience 50 \
#   --seed 42 \
#   --mode puregpu \

python main.py \
  --dataset reddit \
  --model gcn  \
  --sampling NS \
  --dropout 0.3 \
  --lr 0.001 \
  --n-epochs 10 \
  --n-gpus 4 \
  --n-layers 6 \
  --n-hidden 1024 \
  --num-heads 2 \
  --batch-size 4096 \
  --fanout 4 \
  --agg mean \
  --log-every 5 \
  --patience 50 \
  --seed 42 \
  # --mode puregpu \