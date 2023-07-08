python main.py \
  --dataset reddit \
  --model graphsage \
  --sampling NS \
  --dropout 0.3 \
  --lr 0.001 \
  --n-epochs 50 \
  --n-gpus 4 \
  --n-layers 3 \
  --n-hidden 256 \
  --batch-size 1024 \
  --fanout 4 \
  --agg mean \
  --log-every 5 \
#   --seed 10245829 \
#   --mode puregpu \
  # --seed \