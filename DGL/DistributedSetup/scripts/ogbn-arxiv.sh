#!/bin/bash
#SBATCH --job-name=arxiv-dist      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --mem=50G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --partition=cpu
#SBATCH --time=04:00:00          # total run time limit (HH:MM:SS)

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


for n_partitions in 2 4 8 16 32 64; 
do
python3 /work/sbajaj_umass_edu/dgl/tools/launch.py \
--workspace /work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/DGL/DistributedSetup \
--num_trainers $n_partitions \
--num_samplers 0 \
--num_servers 1 \
--part_config /work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/PipeGCN/partitions/ogbn-arxiv-1-metis-vol-trans/ogbn-arxiv-1-metis-vol-trans.json \
--ip_config ip_config.txt \
"/work/sbajaj_umass_edu/GNNEnv/bin/python3 node_classification.py --dataset ogbn-arxiv --graph_name ogbn-arxiv-1-metis-vol-trans --ip_config ip_config.txt --num_epochs 200 --batch_size 1024 --num_hidden 512 --num_layers 2 --fan_out 20,20 --eval_every 2"
done