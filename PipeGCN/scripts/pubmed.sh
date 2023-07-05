python /home/ubuntu/GNN_mini_vs_full/GNN_minibatch_vs_fullbatch/PipeGCN/main.py \
  --dataset pubmed \
  --dropout 0.5 \
  --lr 0.0001 \
  --n-partitions 4 \
  --n-epochs 1000 \
  --model graphsage \
  --n-layers 5 \
  --n-hidden 256 \
  --log-every 5 \
  --use-pp \
  --convergence-threshold 0.0001\
  # --enable-pipeline \
  # --norm layer\
  # --seed 837330801 \
  # --fix-seed \
  # --inductive \
  # --parts-per-node 2 \
  # --backend nccl \
