python train.py \
  --dataset ogbn-products \
  --model GAT \
  --dropout 0.6 \
  --lr 0.0005 \
  --n-epochs 100 \
  --n-layers 3 \
  --n-hidden 128 \
  --batch-size 1024 \
  --fanout 4 \
  --num-heads 7 \
  --agg mean \
  --log-every 5 \
#   --seed 3485963027166655500 \
