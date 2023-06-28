python3 main.py \
  --dataset ogbn-products \
  --dropout 0.3 \
  --lr 0.003 \
  --n-gpus 4 \
  --n-epochs 16 \
  --model graphsage \
  --n-layers 3 \
  --n-hidden 128 \
  --log-every 5 \
  --batch-size 2048 \
  # --use-pp \
  # --enable-pipeline \
  # --fix-seed \
  # --inductive \
  # --parts-per-node 2 \
  # --backend nccl \
