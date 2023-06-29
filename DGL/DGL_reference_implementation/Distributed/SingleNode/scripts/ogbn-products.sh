python3 main.py \
  --dataset ogbn-products \
  --dropout 0.3 \
  --lr 0.003 \
  --n-gpus 4 \
  --n-epochs 6 \
  --model graphsage \
  --log-every 5 \
  --n-layers 3 \
  --n-hidden 128 \
  --batch-size 2048 \
  --fanout 4 \
  --agg mean \
  # --use-pp \
  # --enable-pipeline \
  # --fix-seed \
  # --inductive \
  # --parts-per-node 2 \
  # --backend nccl \
