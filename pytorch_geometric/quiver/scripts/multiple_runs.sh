# python quiver_reddit_pubmed.py \
#   --dataset pubmed \
#   --model graphsage \
#   --sampling NS \
#   --dropout 0.5 \
#   --lr 0.001 \
#   --n-epochs 500 \
#   --n-gpus 4 \
#   --n-layers 2 \
#   --n-hidden 626 \
#   --batch-size 512 \
#   --eval-batch-size 1024 \
#   --weight-decay 5e-4 \
#   --fanout 4 \
#   --agg max \
#   --log-every 5 \
#   --seed 12345 \


python quiver_ogbn.py \
  --dataset ogbn-products \
  --model graphsage \
  --sampling NS \
  --dropout 0.5 \
  --lr 0.01 \
  --n-epochs 101 \
  --n-gpus 4 \
  --n-layers 3 \
  --n-hidden 256 \
  --batch-size 1024 \
  --eval-batch-size 1024 \
  --weight-decay 0 \
  --fanout 5 \
  --agg mean \
  --log-every 5 \
  --seed 12345 \


# python quiver_reddit_pubmed.py \
#   --dataset reddit \
#   --model graphsage \
#   --sampling NS \
#   --dropout 0.5 \
#   --lr 0.001 \
#   --n-epochs 21 \
#   --n-gpus 4 \
#   --n-layers 2 \
#   --n-hidden 256 \
#   --batch-size 1024 \
#   --eval-batch-size 1024 \
#   --weight-decay 0 \
#   --fanout 5 \
#   --agg mean \
#   --log-every 5 \
#   --seed 12345 \


# python quiver_ogbn.py \
#   --dataset ogbn-arxiv \
#   --model graphsage \
#   --sampling NS \
#   --dropout 0.5 \
#   --lr 0.001 \
#   --n-epochs 21 \
#   --n-gpus 4 \
#   --n-layers 2 \
#   --n-hidden 256 \
#   --batch-size 1024 \
#   --eval-batch-size 1024 \
#   --weight-decay 0 \
#   --fanout 5 \
#   --agg mean \
#   --log-every 5 \
#   --seed 12345 \