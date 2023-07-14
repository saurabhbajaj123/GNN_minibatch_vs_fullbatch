python saint_main.py \
  --dataset reddit \
  --model graphsage \
  --sampling saint \
  --dropout 0.3 \
  --lr 0.003 \
  --n-epochs 500 \
  --n-gpus 4 \
  --n-layers 3 \
  --n-hidden 128 \
  --batch-size 1024 \
  --budget_node_edge 256 \
  --budget_rw (256, 512) \
  --agg mean \
  --log-every 5 \
#   --seed 10245829 \
#   --mode puregpu \
  # --seed \