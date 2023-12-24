for n_parts in 1 2 3 4
do
  echo $n_parts
  python3 main.py \
    --dataset ogbn-products \
    --dropout 0.3 \
    --lr 0.007 \
    --n-partitions $n_parts \
    --n-epochs 5 \
    --model graphsage \
    --n-layers 3 \
    --n-hidden 128 \
    --log-every 10 \
    --use-pp \
    --fix-seed \
    --patience 50 \
    --enable-pipeline \
    --seed 7635837650068751000
done
