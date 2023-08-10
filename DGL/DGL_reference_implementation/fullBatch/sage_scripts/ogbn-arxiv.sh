
python3 sage_train.py \
  --dataset ogbn-arxiv \
  --dropout 0.3 \
  --lr 0.001 \
  --n-epochs 2000 \
  --n-layers 4 \
  --n-hidden 1024 \
  --log-every 5 \
  --agg gcn \
  # --seed 42 \