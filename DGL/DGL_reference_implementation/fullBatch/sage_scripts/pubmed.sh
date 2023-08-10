
python3 sage_train.py \
  --dataset pubmed \
  --dropout 0.55 \
  --lr 1e-4 \
  --n-epochs 1000 \
  --n-layers 10 \
  --n-hidden 512 \
  --log-every 5 \
  --agg gcn \
  --device_id 1 \
#   --seed 42 \