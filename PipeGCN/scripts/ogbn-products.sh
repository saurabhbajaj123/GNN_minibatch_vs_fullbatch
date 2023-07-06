python /home/ubuntu/GNN_mini_vs_full/GNN_minibatch_vs_fullbatch/PipeGCN/main.py \
  --dataset ogbn-products \
  --dropout 0.3 \
  --lr 0.003 \
  --n-partitions 4 \
  --n-epochs 500 \
  --model graphsage \
  --n-layers 4 \
  --n-hidden 127 \
  --log-every 5 \
  --use-pp \
  --convergence-threshold 0.0001\
  --fix-seed \
  --seed 837330801 \
  # --norm layer\
  # --enable-pipeline \
  # --inductive \
  # --parts-per-node 2 \
  # --backend nccl \
