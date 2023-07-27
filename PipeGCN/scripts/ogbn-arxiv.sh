python main.py \
  --dataset ogbn-arxiv \
  --dropout 0.3 \
  --lr 0.0006 \
  --n-partitions 4 \
  --n-epochs 2000 \
  --model graphsage \
  --n-layers 3 \
  --n-hidden 128 \
  --log-every 5 \
  --enable-pipeline \
  # --use-pp
