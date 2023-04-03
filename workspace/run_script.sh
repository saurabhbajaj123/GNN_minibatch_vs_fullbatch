python3 ~/workspace/dgl/tools/launch.py \
--workspace ~/workspace/ \
--num_trainers 1 \
--num_samplers 0 \
--num_servers 1 \
--part_config partition_dataset/part_config.json \
--ip_config ip_config.txt \
"python3 train_dist.py"