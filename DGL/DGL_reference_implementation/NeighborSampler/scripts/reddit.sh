python train.py \
  --dataset reddit \
  --model GAT \
  --dropout 0.5 \
  --lr 0.005 \
  --n-epochs 100 \
  --n-layers 5 \
  --n-hidden 64 \
  --batch-size 256 \
  --fanout 8 \
  --num-heads 12 \
  --device_id 0 \
  --agg gcn \
  --log-every 5 \
#   --seed 3485963027166655500 \
