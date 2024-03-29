python train.py \
  --dataset reddit \
  --model GCN \
  --dropout 0.5 \
  --lr 0.0005 \
  --n-epochs 3000 \
  --n-layers 3 \
  --n-hidden 256 \
  --batch-size 1024 \
  --fanout 8 \
  --device_id 3 \
  --num-heads 4 \
  --agg mean \
  --log-every 5 \
  --seed 42 \
