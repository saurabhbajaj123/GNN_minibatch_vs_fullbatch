python main.py \
  --dataset ogbn-arxiv \
  --dropout 0.3 \
  --lr 0.003 \
  --n-partitions 4 \
  --n-epochs 1000 \
  --model graphsage \
  --n-layers 3 \
  --n-hidden 256 \
  --log-every 5 \
  --enable-pipeline \
  --use-pp
