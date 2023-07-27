python main.py \
  --dataset ogbn-products \
  --model graphsage \
  --sampling NS \
  --dropout 0.3 \
  --lr 0.001 \
  --n-epochs 500 \
  --n-gpus 4 \
  --n-layers 6 \
  --n-hidden 256 \
  --batch-size 512 \
  --fanout 4 \
  --agg mean \
  --log-every 5 \
  --seed 42
#   --seed 10245829 \
#   --mode puregpu \
  # --seed \