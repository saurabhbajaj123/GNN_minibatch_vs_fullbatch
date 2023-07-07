python main.py \
  --dataset ogbn-products \
  --dropout 0.3 \
  --lr 0.003 \
  --n-epochs 10 \
  --n-gpus 4 \
  --n-layers 3 \
  --n-hidden 728 \
  --batch-size 1024 \
  --fanout 10 \
  --agg mean \
  --log-every 5 \
#   --mode puregpu \
  # --seed \
