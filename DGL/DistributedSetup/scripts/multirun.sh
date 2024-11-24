#!/bin/bash
#SBATCH --job-name=distmulti      # create a short name for your job
#SBATCH --nodes=4                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --mem=250G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:8             # number of gpus per node
#SBATCH --partition=gpu-preempt
#SBATCH --constraint=2080ti
#SBATCH --time=00:40:00          # total run time limit (HH:MM:SS)

export GLOO_SOCKET_IFNAME=`ip -o -4 route show to default | awk '{print $5}'`

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "MASTER_PORT="$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE


HOSTNAMES=$(scontrol show hostnames )
echo "HOSTNAMES="$HOSTNAMES 
> ip_config.txt
for var in $HOSTNAMES; do
    # echo $var >> ip_config.txt
    nslookup $var 10.10.1.3 | awk '/^Address: / { print $2 }' >> ip_config.txt
done 


master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

source /work/sbajaj_umass_edu/GNNEnv/bin/activate


# for n_partitions in 4 ; 
# do
# python3 /work/sbajaj_umass_edu/dgl/tools/launch.py \
# --workspace /work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/DGL/DistributedSetup \
# --num_trainers $n_partitions \
# --num_samplers 0 \
# --num_servers 1 \
# --part_config /work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/PipeGCN/partitions/pubmed-1-metis-vol-trans/pubmed-1-metis-vol-trans.json \
# --ip_config ip_config.txt \
# "/work/sbajaj_umass_edu/GNNEnv/bin/python3 node_classification.py --dataset pubmed --graph_name pubmed-1-metis-vol-trans --ip_config ip_config.txt --num_epochs 10 --batch_size 1024 --num_hidden 256 --num_layers 3 --fan_out 10,10,10 --eval_every 2"
# done


# # for n_partitions in 2 4 8 16 32 64; 
# # do
# # python3 /work/sbajaj_umass_edu/dgl/tools/launch.py \
# # --workspace /work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/DGL/DistributedSetup \
# # --num_trainers $n_partitions \
# # --num_samplers 0 \
# # --num_servers 1 \
# # --part_config /work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/PipeGCN/partitions/ogbn-arxiv-1-metis-vol-trans/ogbn-arxiv-1-metis-vol-trans.json \
# # --ip_config ip_config.txt \
# # "/work/sbajaj_umass_edu/GNNEnv/bin/python3 node_classification.py --dataset ogbn-arxiv --graph_name ogbn-arxiv-1-metis-vol-trans --ip_config ip_config.txt --num_epochs 200 --batch_size 1024 --num_hidden 512 --num_layers 2 --fan_out 20,20 --eval_every 2"
# # done

# for n_partitions in 64 32 16 8 4 2; 
# do
# python3 /work/sbajaj_umass_edu/dgl/tools/launch.py \
# --workspace /work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/DGL/DistributedSetup \
# --num_trainers $n_partitions \
# --num_samplers 0 \
# --num_servers 1 \
# --part_config /work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/PipeGCN/partitions/reddit-1-metis-vol-trans/reddit-1-metis-vol-trans.json \
# --ip_config ip_config.txt \
# "/work/sbajaj_umass_edu/GNNEnv/bin/python3 node_classification.py --dataset reddit --graph_name reddit-1-metis-vol-trans --ip_config ip_config.txt --num_epochs 200 --batch_size 1024 --num_hidden 1024 --num_layers 4 --fan_out 5,5,5,5 --eval_every 2"
# done


for n_partitions in 8; 
do
python3 /work/sbajaj_umass_edu/dgl/tools/launch.py \
--workspace /work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/DGL/DistributedSetup \
--num_trainers $n_partitions \
--num_samplers 0 \
--num_servers 2 \
--part_config /work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/PipeGCN/partitions/ogbn-products-4-metis-vol-trans/ogbn-products-4-metis-vol-trans.json \
--ip_config ip_config.txt \
"/work/sbajaj_umass_edu/GNNEnv/bin/python3 node_classification.py --dataset ogbn-products --graph_name ogbn-products-4-metis-vol-trans --ip_config ip_config.txt --num_epochs 200 --batch_size 1024 --num_hidden 256 --num_layers 5 --fan_out 5,5,5,5,5 --eval_every 10 --num_gpus 8"
done

# for n_partitions in 2 4 8 16 32 64; do
# python3 /work/sbajaj_umass_edu/dgl/tools/launch.py \
# --workspace /work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/DGL/DistributedSetup \
# --num_trainers $n_partitions \
# --num_samplers 0 \
# --num_servers 1 \
# --part_config /work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/DGL/DistributedSetup/partitions/ogbn-papers100m_frac_100.0_hops_2_subgraph-1-metis-vol-trans/ogbn-papers100m_frac_100.0_hops_2_subgraph-1-metis-vol-trans.json \
# --ip_config ip_config.txt \
# "/work/sbajaj_umass_edu/GNNEnv/bin/python3 node_classification.py --dataset ogbn-papers100m --graph_name ogbn-papers100m_frac_100.0_hops_2_subgraph-1-metis-vol-trans --ip_config ip_config.txt --num_epochs 20 --batch_size 1024 --num_hidden 128 --num_layers 2 --fan_out 5,5 --eval_every 2 --lr 0.003"
# done