python3 main.py \
  --dataset ogbn-products \
  --dropout 0.3 \
  --lr 0.003 \
  --n-gpus 4 \
  --n-epochs 50 \
  --model graphsage \
  --log-every 5 \
  --n-layers 4 \
  --n-hidden 128 \
  --batch-size 1024 \
  --fanout 4 \
  --agg gcn \
  # --use-pp \
  # --enable-pipeline \
  # --fix-seed \
  # --inductive \
  # --parts-per-node 2 \
  # --backend nccl \
