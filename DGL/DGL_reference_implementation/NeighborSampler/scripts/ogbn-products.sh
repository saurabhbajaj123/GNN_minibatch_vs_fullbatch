python train.py \
  --dataset ogbn-products \
  --model GAT \
  --dropout 0.5 \
  --lr 0.003 \
  --n-epochs 1000 \
  --n-layers 2 \
  --n-hidden 256 \
  --batch-size 1024 \
  --fanout 10 \
  --num-heads 10 \
  --agg mean \
  --log-every 5 \
#   --seed 3485963027166655500 \
