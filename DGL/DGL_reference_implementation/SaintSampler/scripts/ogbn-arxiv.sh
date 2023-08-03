python train.py \
  --dataset ogbn-arxiv \
  --model SAGE \
  --n-epochs 50 \
  --n-layers 2 \
  --n-hidden 16 \
  --lr 0.001 \
  --dropout 0.5 \
  --seed 42 \
  --device_id 1 \
  --budget_node_edge 1220 \
  --budget_rw_0 256 \
  --budget_rw_1 16 \
  --mode_saint walk \
  --batch_size 1024 \
  --log_every 5 \