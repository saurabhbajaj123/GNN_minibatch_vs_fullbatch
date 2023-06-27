python /home/ubuntu/GNN_mini_vs_full/GNN_minibatch_vs_fullbatch/PipeGCN/main.py \
  --dataset ogbn-products \
  --dropout 0.3 \
  --lr 0.003 \
  --n-partitions 7 \
  --n-epochs 2000 \
  --model graphsage \
  --n-layers 7 \
  --n-hidden 128 \
  --log-every 5 \
  --use-pp \
  --enable-pipeline \
  # --fix-seed \
  # --inductive \
  # --parts-per-node 2 \
  # --backend nccl \
