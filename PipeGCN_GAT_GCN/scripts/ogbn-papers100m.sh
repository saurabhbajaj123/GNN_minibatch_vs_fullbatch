python /home/ubuntu/GNN_mini_vs_full/GNN_minibatch_vs_fullbatch/PipeGCN/main.py \
  --dataset ogbn-papers100m \
  --dropout 0.3 \
  --lr 0.003 \
  --n-partitions 8 \
  --n-epochs 20 \
  --model graphsage \
  --n-layers 1 \
  --n-hidden 16 \
  --log-every 10 \
  --enable-pipeline \
  --use-pp

  # --partition-method parmetis \