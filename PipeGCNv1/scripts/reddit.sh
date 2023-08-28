python main.py \
  --dataset reddit \
  --dropout 0.5 \
  --lr 0.01 \
  --n-partitions 7 \
  --n-epochs 2000 \
  --model graphsage \
  --n-layers 10 \
  --n-hidden 256 \
  --log-every 5 \
  --enable-pipeline \
  --use-pp

  # --inductive \