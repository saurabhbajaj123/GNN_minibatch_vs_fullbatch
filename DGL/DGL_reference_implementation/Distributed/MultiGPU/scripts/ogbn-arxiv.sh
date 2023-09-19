#!/bin/bash

#SBATCH --job-name ogbn-arxiv   ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --mem=20GB  # memory per CPU core
#SBATCH --time=0-02:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --output=result_arxiv.txt
#SBATCH --constraint=m40
#SBATCH --nodes=1


source /work/sbajaj_umass_edu/GNNEnv/bin/activate

python main.py \
  --dataset ogbn-arxiv \
  --model graphsage \
  --sampling NS \
  --dropout 0.3 \
  --lr 0.003 \
  --n-epochs 1000 \
  --n-gpus 4 \
  --n-layers 4 \
  --n-hidden 859 \
  --batch-size 1024 \
  --patience 50 \
  --fanout 4 \
  --agg mean \
  --log-every 5 \
  --seed 42 \
  --mode puregpu \

# dropout, lr, fanout, batch_size
