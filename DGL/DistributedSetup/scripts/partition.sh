#!/bin/bash

#SBATCH --job-name partition  ## name that will show up in the queue
#SBATCH --mem=500GB  # memory per CPU core
#SBATCH --partition=cpu
#SBATCH --time=0-23:59:00  ## time for analysis (day-hour:min:sec)

source /home/ubuntu/gnn_mini_vs_full/GNNEnv/bin/activate

python3 partition_graph.py \
  --dataset ogbn-papers100m_frac_100.0_hops_2_subgraph \
  --subgraph-dataset /home/ubuntu/gnn_mini_vs_full/GNN_minibatch_vs_fullbatch/DGL/DGL_reference_implementation/subgraph/ogbn-papers100M_frac_100.0_hops_2_subgraph.bin \
  --num_parts 1 \
  --part_method metis \
  # --output /home/ubuntu/gnn_mini_vs_full/GNN_minibatch_vs_fullbatch/DGL/DistributedSetup/partitions/ogbn-papers100M_frac_100.0_hops_2_subgraph_12_parts \
#   --balance_train \
#   --undirected \
#   --balance_edges \

python3 partition_graph.py \
  --dataset ogbn-arxiv \
  --num_parts 1 \
  --output /home/ubuntu/gnn_mini_vs_full/GNN_minibatch_vs_fullbatch/DGL/DistributedSetup/ogbn-arxiv-1-metis-vol-trans \
# #   --balance_train \
# #   --undirected \
# #   --balance_edges \

# python3 partition_graph.py \
#   --dataset ogbn-papers100M_frac_100.0_hops_3_subgraph \
#   --subgraph-dataset /home/ubuntu/gnn_mini_vs_full/GNN_minibatch_vs_fullbatch/dataset/papers_subgraphs/ogbn-papers100M_frac_100.0_hops_3_subgraph.bin \
#   --num_parts 3 \
#   --output /home/ubuntu/gnn_mini_vs_full/GNN_minibatch_vs_fullbatch/DGL/DistributedSetup/partitions/ogbn-papers100M_frac_100.0_hops_3_subgraph_3_parts \

# python3 partition_graph.py \
#   --dataset ogbn-papers100M_frac_100.0_hops_4_subgraph \
#   --subgraph-dataset /home/ubuntu/gnn_mini_vs_full/GNN_minibatch_vs_fullbatch/dataset/papers_subgraphs/ogbn-papers100M_frac_100.0_hops_4_subgraph.bin \
#   --num_parts 3 \
#   --output /home/ubuntu/gnn_mini_vs_full/GNN_minibatch_vs_fullbatch/DGL/DistributedSetup/partitions/ogbn-papers100M_frac_100.0_hops_4_subgraph_3_parts \

