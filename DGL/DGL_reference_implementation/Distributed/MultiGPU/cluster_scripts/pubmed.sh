python cluster_main.py \
  --dataset pubmed \
  --model graphsage \
  --sampling cluster \
  --dropout 0.3 \
  --lr 0.003 \
  --n-epochs 1000 \
  --n-gpus 4 \
  --n-layers 3 \
  --n-hidden 256 \
  --batch-size 1024 \
  --num-partitions 2000 \
  --agg mean \
  --log-every 5 \
  --seed 6238418958544123000 \
#   --mode puregpu \
  # --seed \