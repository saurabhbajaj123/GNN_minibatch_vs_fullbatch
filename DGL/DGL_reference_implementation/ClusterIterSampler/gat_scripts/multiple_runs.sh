# python gat_train.py \
#   --dataset ogbn-products \
#   --n-epochs 20 \
#   --lr 0.004 \
#   --dropout 0.5 \
#   --batch_size 1024 \
#   --n-layers 2 \
#   --n-hidden 128 \
#   --num-heads 2 \
#   --num_partitions 6000 \
#   --device_id 0 \
#   --seed 42 \
#   --log_every 5 \

# python gat_train.py \
#   --dataset ogbn-arxiv \
#   --n-epochs 20 \
#   --lr 0.004 \
#   --dropout 0.5 \
#   --batch_size 256 \
#   --n-layers 2 \
#   --n-hidden 128 \
#   --num-heads 2 \
#   --num_partitions 1000 \
#   --device_id 0 \
#   --seed 42 \
#   --log_every 5 \


python gat_train.py \
  --dataset reddit \
  --n-epochs 2000 \
  --lr 0.003 \
  --dropout 0.5 \
  --batch_size 1024 \
  --n-layers 3 \
  --n-hidden 256 \
  --num-heads 12 \
  --num_partitions 4000 \
  --device_id 1 \
  --seed 42 \
  --log_every 5 \


# python gat_train.py \
#   --dataset pubmed \
#   --n-epochs 200 \
#   --lr 0.004 \
#   --dropout 0.5 \
#   --batch_size 256 \
#   --n-layers 2 \
#   --n-hidden 128 \
#   --num-heads 2 \
#   --num_partitions 1000 \
#   --device_id 0 \
#   --seed 42 \
#   --log_every 5 \
