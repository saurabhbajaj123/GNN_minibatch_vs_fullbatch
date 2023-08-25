# #vanilla
# python main.py \
#   --dataset ogbn-arxiv \
#   --dropout 0.3 \
#   --lr 0.003 \
#   --n-partitions 4 \
#   --n-epochs 2000 \
#   --model graphsage \
#   --n-layers 4 \
#   --n-hidden 859 \
#   --log-every 5 \
#   --fix-seed \
#   --seed 1261325436 \

# python /home/ubuntu/GNN_mini_vs_full/GNN_minibatch_vs_fullbatch/PipeGCN/main.py \
#   --dataset pubmed \
#   --dropout 0.7565688403188127 \
#   --lr 0.0001 \
#   --n-partitions 4 \
#   --n-epochs 1000 \
#   --model graphsage \
#   --n-layers 3 \
#   --n-hidden 187 \
#   --log-every 5 \
#   --use-pp \
#   --fix-seed \
#   --seed 1344439319 \

# python /home/ubuntu/GNN_mini_vs_full/GNN_minibatch_vs_fullbatch/PipeGCN/main.py \
#   --dataset ogbn-products \
#   --dropout 0.3 \
#   --lr 0.003 \
#   --n-partitions 4 \
#   --n-epochs 500 \
#   --model graphsage \
#   --n-layers 4 \
#   --n-hidden 127 \
#   --log-every 5 \
#   --use-pp \
#   --fix-seed \
#   --seed 837330801 \

# python /home/ubuntu/GNN_mini_vs_full/GNN_minibatch_vs_fullbatch/PipeGCN/main.py \
#   --dataset reddit \
#   --dropout 0.3 \
#   --lr 0.001 \
#   --n-partitions 4 \
#   --n-epochs 1000 \
#   --model graphsage \
#   --n-layers 4 \
#   --n-hidden 312 \
#   --log-every 5 \
#   --fix-seed \
#   --seed 1586505639 \
#   --use-pp \



# # enable pipeline on
# python main.py \
#   --dataset ogbn-arxiv \
#   --dropout 0.3 \
#   --lr 0.003 \
#   --n-partitions 4 \
#   --n-epochs 2000 \
#   --model graphsage \
#   --n-layers 4 \
#   --n-hidden 859 \
#   --log-every 5 \
#   --fix-seed \
#   --seed 1261325436 \
#   --enable-pipeline \


python /home/ubuntu/GNN_mini_vs_full/GNN_minibatch_vs_fullbatch/PipeGCN/main.py \
  --dataset pubmed \
  --dropout 0.7565688403188127 \
  --lr 0.0001 \
  --n-partitions 4 \
  --n-epochs 1000 \
  --model graphsage \
  --n-layers 3 \
  --n-hidden 187 \
  --log-every 5 \
  --use-pp \
  --fix-seed \
  --seed 1344439319 \
  --enable-pipeline \


# python /home/ubuntu/GNN_mini_vs_full/GNN_minibatch_vs_fullbatch/PipeGCN/main.py \
#   --dataset ogbn-products \
#   --dropout 0.3 \
#   --lr 0.001 \
#   --n-partitions 4 \
#   --n-epochs 1000 \
#   --model graphsage \
#   --n-layers 4 \
#   --n-hidden 127 \
#   --log-every 5 \
#   --use-pp \
#   --fix-seed \
#   --seed 22978128 \
#   --enable-pipeline \

# python /home/ubuntu/GNN_mini_vs_full/GNN_minibatch_vs_fullbatch/PipeGCN/main.py \
#   --dataset reddit \
#   --dropout 0.3 \
#   --lr 0.001 \
#   --n-partitions 4 \
#   --n-epochs 1000 \
#   --model graphsage \
#   --n-layers 4 \
#   --n-hidden 312 \
#   --log-every 5 \
#   --fix-seed \
#   --seed 1586505639 \
#   --enable-pipeline \
