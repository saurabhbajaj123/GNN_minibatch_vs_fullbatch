python train.py \
  --dataset pubmed \
  --model GCN \
  --dropout 0.5 \
  --lr 0.0005 \
  --device_id 1 \
  --n-epochs 1000 \
  --n-layers 7 \
  --n-hidden 329 \
  --batch-size 1024 \
  --fanout 9 \
  --num-heads 7 \
  --agg mean \
  --log-every 5 \
  # --seed 3485963027166655500 \
