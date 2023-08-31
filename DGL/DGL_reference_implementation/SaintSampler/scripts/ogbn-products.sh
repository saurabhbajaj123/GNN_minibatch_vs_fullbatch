
python testing_new_saint_class.py \
  --dataset ogbn-products \
  --model SAGE \
  --n-epochs 500 \
  --n-layers 2 \
  --n-hidden 16 \
  --lr 0.001 \
  --dropout 0.5 \
  --seed 42 \
  --device_id 2 \
  --num-iters 1000 \
  --budget_node_edge 6000 \
  --budget_rw_0 256 \
  --budget_rw_1 16 \
  --mode_saint node \
  --batch_size 1024 \
  --log_every 5 \
