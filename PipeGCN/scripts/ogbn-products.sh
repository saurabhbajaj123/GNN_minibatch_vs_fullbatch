python /home/ubuntu/GNN_mini_vs_full/GNN_minibatch_vs_fullbatch/PipeGCN/main.py \
  --dataset ogbn-products \
  --dropout 0.3 \
  --lr 0.003 \
  --n-partitions 7 \
  --n-epochs 10 \
  --model graphsage \
  --n-layers 2 \
  --n-hidden 16 \
  --log-every 5 \
  --use-pp \
  --fix-seed \
  --enable-pipeline \
  --inductive \
  # --parts-per-node 2 \
  # --backend nccl \
