python main.py \
  --dataset ogbn-arxiv \
  --model graphsage \
  --sampling NS \
  --dropout 0.3 \
  --lr 0.003 \
  --n-epochs 1000 \
  --n-gpus 4 \
  --n-layers 3 \
  --n-hidden 256 \
  --batch-size 1024 \
  --fanout 4 \
  --agg mean \
  --log-every 5 \
  --seed 42 \
