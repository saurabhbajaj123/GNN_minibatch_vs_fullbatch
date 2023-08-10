python3 gcn_train.py \
  --dataset ogbn-arxiv \
  --dropout 0.5 \
  --lr 0.001 \
  --n-epochs 3000 \
  --n-layers 5 \
  --n-hidden 728 \
  --log-every 5 \
  --seed 42 \

# python3 gcn_train.py \
#   --dataset reddit \
#   --dropout 0.5 \
#   --lr 0.0015 \
#   --n-epochs 1000 \
#   --n-layers  2 \
#   --n-hidden 512 \
#   --log-every 5 \
#   --seed 42 \


# python3 gcn_train.py \
#   --dataset ogbn-products \
#   --dropout 0.5 \
#   --lr 0.0015 \
#   --n-epochs 500 \
#   --n-layers 5 \
#   --n-hidden 64 \
#   --log-every 5 \
#   --seed 42 \