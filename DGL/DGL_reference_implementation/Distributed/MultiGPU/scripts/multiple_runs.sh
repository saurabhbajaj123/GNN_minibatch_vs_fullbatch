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

# for ((gpus = 1; gpus <= 4; gpus++))
# do 
# python main.py \
#   --dataset ogbn-arxiv \
#   --model graphsage \
#   --sampling NS \
#   --dropout 0.3 \
#   --lr 0.001 \
#   --n-epochs 1000 \
#   --n-gpus 4 \
#   --n-layers 3 \
#   --n-hidden 256 \
#   --batch-size 1024 \
#   --fanout 4 \
#   --agg mean \
#   --log-every 5 \
#   --seed 42
# done

# for ((gpus = 1; gpus <= 4; gpus++))
# do 
# python main.py \
#   --dataset ogbn-products \
#   --model graphsage \
#   --sampling NS \
#   --dropout 0.3 \
#   --lr 0.001 \
#   --n-epochs 500 \
#   --n-gpus 4 \
#   --n-layers 3 \
#   --n-hidden 256 \
#   --batch-size 2048 \
#   --fanout 4 \
#   --agg mean \
#   --log-every 5 \
#   --seed 42 
# done

# for ((gpus = 1; gpus <= 4; gpus++))
# do 
python main.py \
  --dataset pubmed \
  --model graphsage \
  --sampling NS \
  --dropout 0.3 \
  --lr 0.001 \
  --n-epochs 100 \
  --n-gpus 4 \
  --n-layers 4 \
  --n-hidden 256 \
  --batch-size 2048 \
  --fanout 10 \
  --agg mean \
  --log-every 5 \
  --seed 42
# done

# for ((gpus = 4; gpus >= 1; gpus--))
# do 
#   python main.py \
#     --dataset reddit \
#     --model graphsage \
#     --sampling NS \
#     --dropout 0.3 \
#     --lr 0.001 \
#     --n-epochs 500 \
#     --n-gpus $gpus \
#     --n-layers 3 \
#     --n-hidden 256 \
#     --batch-size 2048 \
#     --fanout 4 \
#     --agg mean \
#     --log-every 5 \
#     --seed 42
# done