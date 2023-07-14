python saint_main.py \
  --dataset ogbn-products \
  --model graphsage \
  --sampling saint \
  --dropout 0.3 \
  --lr 0.003 \
  --n-epochs 500 \
  --n-gpus 4 \
  --n-layers 3 \
  --n-hidden 128 \
  --batch-size 1024 \
  --budget_node_edge 1220 \
  --budget_rw_0 256 \
  --budget_rw_1 16 \
  --mode_saint edge \
  --log-every 5 \
#   --seed 10245829 \
#   --mode puregpu \
  # --seed \


  