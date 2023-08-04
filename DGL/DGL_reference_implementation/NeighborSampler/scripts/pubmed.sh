python train.py \
  --dataset pubmed \
  --model GAT \
  --dropout 0.5 \
  --lr 0.0005 \
  --n-epochs 200 \
  --n-layers 7 \
  --n-hidden 329 \
  --batch-size 1024 \
  --fanout 9 \
  --num-heads 7 \
  --agg mean \
  --log-every 5 \
  # --seed 3485963027166655500 \
