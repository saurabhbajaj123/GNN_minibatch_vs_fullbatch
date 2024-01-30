#!/bin/bash

#SBATCH --job-name orkutdgl   ## name that will show up in the queue
#SBATCH --gpus=4
#SBATCH --mem=100GB  # memory per CPU core
#SBATCH --time=0-04:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --partition=gypsum-m40
#SBATCH --nodes=1


source /work/sbajaj_umass_edu/GNNEnv/bin/activate

# for n_gpus in 4
# do 
python main.py \
  --dataset orkut \
  --model graphsage \
  --sampling NS \
  --dropout 0.5 \
  --lr 0.0005 \
  --n-epochs 5 \
  --n-gpus 1 \
  --n-layers 2 \
  --n-hidden 128 \
  --num-heads 1 \
  --batch-size 1024 \
  --patience 50 \
  --fanout 10 \
  --agg mean \
  --log-every 4 \
  --seed 42 \
  # --mode puregpu
# done