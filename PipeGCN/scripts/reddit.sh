python /home/ubuntu/GNN_mini_vs_full/GNN_minibatch_vs_fullbatch/PipeGCN/main.py \
  --dataset reddit \
  --dropout 0.5 \
  --lr 0.01 \
  --n-partitions 4 \
  --n-epochs 3000 \
  --model graphsage \
  --n-layers 4 \
  --n-hidden 256 \
  --log-every 5 \
  --enable-pipeline \
  --use-pp

  # --inductive \