python cluster_main.py \
  --dataset pubmed \
  --model graphsage \
  --sampling cluster \
  --dropout 0.3 \
  --lr 0.003 \
  --n-epochs 100 \
  --n-gpus 4 \
  --n-layers 3 \
  --n-hidden 128 \
  --batch-size 1024 \
  --num-partitions 5000 \
  --agg mean \
  --log-every 5 \
#   --seed 10245829 \
#   --mode puregpu \
  # --seed \