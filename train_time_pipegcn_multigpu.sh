# #vanilla
# python /home/ubuntu/GNN_mini_vs_full/GNN_minibatch_vs_fullbatch/PipeGCN/main.py \
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
#   --use-pp \
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
# python /home/ubuntu/GNN_mini_vs_full/GNN_minibatch_vs_fullbatch/PipeGCN/main.py \
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
#   --enable-pipeline \


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



# for ((gpus = 1; gpus <= 4; gpus++))
# do 
python /home/ubuntu/GNN_mini_vs_full/GNN_minibatch_vs_fullbatch/DGL/DGL_reference_implementation/Distributed/MultiGPU/main.py \
  --dataset ogbn-arxiv \
  --model graphsage \
  --sampling NS \
  --dropout 0.3 \
  --lr 0.0005 \
  --n-epochs 1000 \
  --n-gpus 4 \
  --n-layers 9 \
  --n-hidden 256 \
  --batch-size 1024 \
  --fanout 4 \
  --agg mean \
  --log-every 5 \
  --seed 42
# done

# for ((gpus = 1; gpus <= 4; gpus++))
# do 
python /home/ubuntu/GNN_mini_vs_full/GNN_minibatch_vs_fullbatch/DGL/DGL_reference_implementation/Distributed/MultiGPU/main.py \
  --dataset ogbn-products \
  --model graphsage \
  --sampling NS \
  --dropout 0.3 \
  --lr 0.0005 \
  --n-epochs 500 \
  --n-gpus 4 \
  --n-layers 5 \
  --n-hidden 256 \
  --batch-size 512 \
  --fanout 4 \
  --agg mean \
  --log-every 5 \
  --seed 42 
# done

# for ((gpus = 1; gpus <= 4; gpus++))
# do 
python /home/ubuntu/GNN_mini_vs_full/GNN_minibatch_vs_fullbatch/DGL/DGL_reference_implementation/Distributed/MultiGPU/main.py \
  --dataset pubmed \
  --model graphsage \
  --sampling NS \
  --dropout 0.3 \
  --lr 0.01 \
  --n-epochs 300 \
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
python /home/ubuntu/GNN_mini_vs_full/GNN_minibatch_vs_fullbatch/DGL/DGL_reference_implementation/Distributed/MultiGPU/main.py \
  --dataset reddit \
  --model graphsage \
  --sampling NS \
  --dropout 0.3 \
  --lr 0.001 \
  --n-epochs 500 \
  --n-gpus 4 \
  --n-layers 7 \
  --n-hidden 256 \
  --batch-size 1024 \
  --fanout 4 \
  --agg mean \
  --log-every 5 \
  --seed 42
# done