python saint_main.py \
  --dataset ogbn-arxiv \
  --model graphsage \
  --sampling saint \
  --dropout 0.3 \
  --lr 0.001 \
  --n-epochs 200 \
  --n-gpus 4 \
  --n-layers 4 \
  --n-hidden 959 \
  --batch-size 1558 \
  --budget_node_edge 1220 \
  --budget_rw_0 256 \
  --budget_rw_1 16 \
  --mode_saint node \
  --log-every 5 \
  --seed 6260732369359939000 \

# python saint_main.py \
#   --dataset pubmed \
#   --model graphsage \
#   --sampling saint \
#   --dropout 0.3 \
#   --lr 0.003 \
#   --n-epochs 100 \
#   --n-gpus 4 \
#   --n-layers 3 \
#   --n-hidden 128 \
#   --batch-size 256 \
#   --budget_node_edge 1024 \
#   --budget_rw_0 256 \
#   --budget_rw_1 16 \
#   --mode_saint node \
#   --log-every 5 \
# #   --seed 10245829 \


# python saint_main.py \
#   --dataset reddit \
#   --model graphsage \
#   --sampling saint \
#   --dropout 0.3 \
#   --lr 0.003 \
#   --n-epochs 500 \
#   --n-gpus 4 \
#   --n-layers 3 \
#   --n-hidden 128 \
#   --batch-size 1024 \
#   --budget_node_edge 256 \
#   --budget_rw_0 256 \
#   --budget_rw_1 16 \
#   --agg mean \
#   --log-every 5 \
# #   --seed 10245829 \



