python cluster_main.py \
  --dataset reddit \
  --model graphsage \
  --sampling cluster \
  --dropout 0.3 \
  --lr 0.003 \
  --n-epochs 1000 \
  --n-gpus 4 \
  --n-layers 4 \
  --n-hidden 512 \
  --batch-size 1024 \
  --num-partitions 8000 \
  --agg mean \
  --log-every 5 \
  --seed 42 \
#   --mode puregpu \
  # --seed \