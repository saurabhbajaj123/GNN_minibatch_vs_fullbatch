python cluster_main.py \
  --dataset ogbn-products \
  --model gcn \
  --sampling cluster \
  --dropout 0.2 \
  --lr 0.0005 \
  --n-epochs 300 \
  --n-gpus 4 \
  --n-layers 7 \
  --n-hidden 706 \
  --batch-size 128 \
  --num-partitions 2000 \
  --agg mean \
  --log-every 5 \
#   --seed 3485963027166655500 \
#   --mode puregpu \
  # --seed \


# python cluster_main.py \
#   --dataset reddit \
#   --model gcn \
#   --sampling cluster \
#   --dropout 0.5 \
#   --lr 0.0007 \
#   --n-epochs 2000 \
#   --n-gpus 4 \
#   --n-layers 3 \
#   --n-hidden 1024 \
#   --batch-size 1024 \
#   --num-partitions 6000 \
#   --agg mean \
#   --log-every 5 \
#   --seed 42 \
# #   --mode puregpu \
#   # --seed \

# python cluster_main.py \
#   --dataset ogbn-arxiv \
#   --model gcn \
#   --sampling cluster \
#   --dropout 0.5 \
#   --lr 0.003 \
#   --n-epochs 1000 \
#   --n-gpus 4 \
#   --n-layers 6 \
#   --n-hidden 512 \
#   --batch-size 1024 \
#   --num-partitions 4000 \
#   --agg mean \
#   --log-every 5 \
# #   --seed 10245829 \
# #   --mode puregpu \
#   # --seed \


# python cluster_main.py \
#   --dataset pubmed \
#   --model gcn \
#   --sampling cluster \
#   --dropout 0.5 \
#   --lr 0.0005 \
#   --n-epochs 1000 \
#   --n-gpus 4 \
#   --n-layers 4 \
#   --n-hidden 512 \
#   --batch-size 1024 \
#   --num-partitions 2000 \
#   --agg mean \
#   --log-every 5 \
# #   --seed 10245829 \
# #   --mode puregpu \
#   # --seed \