python cluster_main.py \
  --dataset ogbn-arxiv \
  --model gat \
  --sampling cluster \
  --dropout 0.5 \
  --lr 0.003 \
  --n-epochs 1000 \
  --n-gpus 4 \
  --n-layers 4 \
  --n-hidden 128 \
  --batch-size 1024 \
  --num-partitions 2000 \
  --num-heads 2 \
  --agg mean \
  --log-every 5 \
#   --seed 10245829 \
#   --mode puregpu \
  # --seed \


# python cluster_main.py \
#   --dataset pubmed \
#   --model gat \
#   --sampling cluster \
#   --dropout 0.5 \
#   --lr 0.0005 \
#   --n-epochs 1000 \
#   --n-gpus 4 \
#   --n-layers 2 \
#   --n-hidden 256 \
#   --num-heads  6\
#   --batch-size 1024 \
#   --num-partitions 1000 \
#   --agg mean \
#   --log-every 5 \
#   --seed 42 \
# #   --mode puregpu \
#   # --seed \


# python cluster_main.py \
#   --dataset ogbn-products \
#   --model gat \
#   --sampling cluster \
#   --dropout 0.5 \
#   --lr 0.005 \
#   --n-epochs 100 \
#   --n-gpus 4 \
#   --n-layers 2 \
#   --n-hidden 256 \
#   --num-heads 10 \
#   --batch-size 512 \
#   --num-partitions 12000 \
#   --agg mean \
#   --log-every 5 \
# #   --seed 3485963027166655500 \
# #   --mode puregpu \
#   # --seed \


# python cluster_main.py \
#   --dataset reddit \
#   --model gat \
#   --sampling cluster \
#   --dropout 0.5 \
#   --lr 0.0005 \
#   --n-epochs 200 \
#   --n-gpus 4 \
#   --n-layers 5 \
#   --n-hidden 256 \
#   --num-heads 15 \
#   --batch-size 1024 \
#   --num-partitions 12000 \
#   --agg mean \
#   --log-every 5 \
#   --seed 42 \
# #   --mode puregpu \
#   # --seed \

