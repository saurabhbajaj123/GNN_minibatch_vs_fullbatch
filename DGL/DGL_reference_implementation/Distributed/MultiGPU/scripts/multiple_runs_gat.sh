# python main.py \
#   --dataset ogbn-arxiv \
#   --model graphsage \
#   --sampling NS \
#   --dropout 0.3 \
#   --lr 0.003 \
#   --n-epochs 1000 \
#   --n-gpus 4 \
#   --n-layers 3 \
#   --n-hidden 256 \
#   --batch-size 1024 \
#   --fanout 4 \
#   --agg mean \
#   --log-every 5 \
#   --seed 42 \

# python main.py \
#   --dataset ogbn-products \
#   --model graphsage \
#   --sampling NS \
#   --dropout 0.3 \
#   --lr 0.003 \
#   --n-epochs 500 \
#   --n-gpus 4 \
#   --n-layers 3 \
#   --n-hidden 256 \
#   --batch-size 512 \
#   --fanout 4 \
#   --agg mean \
#   --log-every 5 \
#   --seed 42 \


# python main.py \
#   --dataset pubmed \
#   --model graphsage \
#   --sampling NS \
#   --dropout 0.3 \
#   --lr 0.0001 \
#   --n-epochs 1000 \
#   --n-gpus 4 \
#   --n-layers 4 \
#   --n-hidden 256 \
#   --batch-size 512 \
#   --fanout 10 \
#   --agg mean \
#   --log-every 5 \
#   --seed 42 \


# python main.py \
#   --dataset reddit \
#   --model graphsage \
#   --sampling NS \
#   --dropout 0.3 \
#   --lr 0.001 \
#   --n-epochs 3000 \
#   --n-gpus 4 \
#   --n-layers 3 \
#   --n-hidden 256 \
#   --batch-size 1024 \
#   --fanout 4 \
#   --agg mean \
#   --log-every 5 \
#   --seed 42 \

# running all with 0.001 lr


python main.py \
  --dataset pubmed \
  --model gat \
  --sampling NS \
  --dropout 0.3 \
  --lr 0.0005 \
  --n-epochs 200 \
  --n-gpus 4 \
  --n-layers 4 \
  --n-hidden 256 \
  --num-heads 2 \
  --batch-size 2048 \
  --fanout 7 \
  --agg mean \
  --log-every 5 \
  --seed 42


python main.py \
  --dataset ogbn-arxiv \
  --model gat \
  --sampling NS \
  --dropout 0.3 \
  --lr 0.0005 \
  --n-epochs 200 \
  --n-gpus 4 \
  --n-layers 3 \
  --n-hidden 256 \
  --num-heads 2 \
  --batch-size 2048 \
  --fanout 15 \
  --agg mean \
  --log-every 5 \
  --seed 42

python main.py \
  --dataset reddit \
  --model gat \
  --sampling NS \
  --dropout 0.3 \
  --lr 0.0005 \
  --n-epochs 200 \
  --n-gpus 4 \
  --n-layers 3 \
  --n-hidden 256 \
  --num-heads 2 \
  --batch-size 2048 \
  --fanout 10 \
  --agg mean \
  --log-every 5 \
  --seed 42

python main.py \
  --dataset ogbn-products \
  --model gat \
  --sampling NS \
  --dropout 0.3 \
  --lr 0.0005 \
  --n-epochs 200 \
  --n-gpus 4 \
  --n-layers 3 \
  --n-hidden 256 \
  --num-heads 2 \
  --batch-size 2048 \
  --fanout 4 \
  --agg mean \
  --log-every 5 \
  --seed 42 