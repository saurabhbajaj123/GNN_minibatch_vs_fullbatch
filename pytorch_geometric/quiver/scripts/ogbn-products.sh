python quiver_ogbn.py \
  --dataset ogbn-products \
  --model graphsage \
  --sampling NS \
  --dropout 0.5 \
  --lr 0.01 \
  --n-epochs 1001 \
  --n-gpus 4 \
  --n-layers 3 \
  --n-hidden 256 \
  --batch-size 1024 \
  --fanout 10 \
  --agg mean \
  --log-every 5 \
#   --seed 10245829 \
#   --mode puregpu \
  # --seed \