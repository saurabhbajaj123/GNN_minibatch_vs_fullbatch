python main.py \
  --dataset reddit \
  --dropout 0.3 \
  --lr 0.001 \
  --n-partitions 4 \
  --n-epochs 1000 \
  --model graphsage \
  --n-layers 4 \
  --n-hidden 312 \
  --log-every 5 \
  --fix-seed \
  --seed 1586505639 \
  --use-pp \
  # --enable-pipeline \
  # --inductive \