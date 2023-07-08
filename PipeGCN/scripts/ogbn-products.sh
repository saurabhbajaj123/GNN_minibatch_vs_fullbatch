python /home/ubuntu/GNN_mini_vs_full/GNN_minibatch_vs_fullbatch/PipeGCN/main.py \
  --dataset ogbn-products \
  --dropout 0.3 \
  --lr 0.001 \
  --n-partitions 4 \
  --n-epochs 1000 \
  --model graphsage \
  --n-layers 4 \
  --n-hidden 127 \
  --log-every 5 \
  --use-pp \
  --convergence-threshold 0.0001\
  --fix-seed \
  --seed 22978128 \
  --enable-pipeline \
  # --norm layer\
  # --inductive \
  # --parts-per-node 2 \
  # --backend nccl \
