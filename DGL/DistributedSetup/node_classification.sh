
# source /work/sbajaj_umass_edu/GNNEnv/bin/activate

cd /work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/DGL
python3 /work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/DGL/dgl/tools/launch.py \
--workspace /work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/DGL/dgl/examples/pytorch/graphsage/dist/ \
--num_trainers 1 \
--num_samplers 0 \
--num_servers 1 \
--part_config /work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/PipeGCN/partitions/ogbn-products-1-metis-vol-trans/ogbn-products-1-metis-vol-trans.json \
--ip_config /work/sbajaj_umass_edu/ip_config.txt \
"source /work/sbajaj_umass_edu/GNNEnv/bin/activate; python3 /work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/DGL/dgl/examples/distributed/graphsage/node_classification.py --graph_name ogbn-products --ip_config /work/sbajaj_umass_edu/ip_config.txt --num_epochs 30 --batch_size 1024 "