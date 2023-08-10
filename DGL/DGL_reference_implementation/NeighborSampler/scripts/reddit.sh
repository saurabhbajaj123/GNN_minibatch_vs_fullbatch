python train.py \
  --dataset reddit \
  --model GAT \
  --dropout 0.5 \
  --lr 0.003 \
  --n-epochs 200 \
  --n-layers 3 \
  --n-hidden 256 \
  --batch-size 1024 \
  --fanout 4 \
  --num-heads 7 \
  --agg mean \
  --log-every 5 \
#   --seed 3485963027166655500 \
