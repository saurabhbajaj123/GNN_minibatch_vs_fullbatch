python /home/ubuntu/GNN_mini_vs_full/GNN_minibatch_vs_fullbatch/PipeGCN/main.py \
  --dataset pubmed \
  --dropout 0.5 \
  --lr 0.0001 \
  --n-partitions 4 \
  --n-epochs 1000 \
  --model graphsage \
  --n-layers 3 \
  --n-hidden 187 \
  --log-every 5 \
  --use-pp \
  --enable-pipeline \
  # --convergence-threshold 1e-8 \
  # --norm layer\
  # --seed 837330801 \
  # --fix-seed \
  # --inductive \
  # --parts-per-node 2 \
  # --backend nccl \
