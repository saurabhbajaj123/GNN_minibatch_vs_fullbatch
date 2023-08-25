python train.py \
  --dataset ogbn-arxiv \
  --model SAGE \
  --dropout 0.6 \
  --lr 0.001 \
  --device_id 2 \
  --n-epochs 250 \
  --n-layers 6 \
  --n-hidden 512 \
  --batch-size 1024 \
  --fanout 9 \
  --num-heads 7 \
  --agg gcn \
  --log-every 5 \
  # --seed 3485963027166655500 \