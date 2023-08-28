python quiver_ogbn.py \
  --dataset ogbn-products \
  --model graphsage \
  --sampling NS \
  --dropout 0.5 \
  --lr 0.01 \
  --n-epochs 101 \
  --n-gpus 4 \
  --n-layers 3 \
  --n-hidden 256 \
  --batch-size 4096 \
  --eval-batch-size 4096 \
  --weight-decay 0 \
  --fanout 5 \
  --agg mean \
  --log-every 5 \
  --seed 12345 \