#!/bin/bash
#SBATCH --job-name=ddp-test      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=32G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:4             # number of gpus per node
#SBATCH --partition=gpu-preempt
#SBATCH --constraint=m40
#SBATCH --time=02:00:00          # total run time limit (HH:MM:SS)

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


python3 /work/sbajaj_umass_edu/dgl/tools/launch.py \
--workspace /work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/DGL/DistributedSetup \
--num_trainers 4 \
--num_samplers 0 \
--num_servers 1 \
--part_config /work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/PipeGCN/partitions/ogbn-arxiv-1-metis-vol-trans/ogbn-arxiv-1-metis-vol-trans.json \
--ip_config ip_config.txt \
"/work/sbajaj_umass_edu/GNNEnv/bin/python3 node_classification.py --dataset ogbn-arxiv --graph_name ogbn-arxiv-1-metis-vol-trans --ip_config ip_config.txt --num_epochs 100 --batch_size 1000 --num_gpus 4 --num_hidden 16 --num_layers 2 --fan_out 5,5"
# "/work/sbajaj_umass_edu/GNNEnv/bin/python3 subprocess_file.py"
# "python3 node_classification.py"
# "python3 process_init_group.py"
# "source /work/sbajaj_umass_edu/GNNEnv/bin/activate python3 process_init_group.py --graph_name ogbn-products --ip_config ip_config.txt --num_epochs 30 --batch_size 1000 --local_rank 0"
# "source /work/sbajaj_umass_edu/GNNEnv/bin/activate python3 node_classification.py --graph_name ogbn-products --ip_config ip_config.txt --num_epochs 30 --batch_size 1000 --local_rank 0"
