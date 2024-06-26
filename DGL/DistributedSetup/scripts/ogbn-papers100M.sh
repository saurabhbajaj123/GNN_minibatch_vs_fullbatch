#!/bin/bash
#SBATCH --job-name=distpaper      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --cpus-per-task=112        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=250G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:4             # number of gpus per node
#SBATCH --partition=gpu-preempt
#SBATCH --constraint=intel8480
#SBATCH --time=00:04:00          # total run time limit (HH:MM:SS)

export GLOO_SOCKET_IFNAME=`ip -o -4 route show to default | awk '{print $5}'`


echo "GLOO_SOCKET_IFNAME="$GLOO_SOCKET_IFNAME
nvidia-smi --query-gpu=gpu_name --format=csv,noheader

nvidia-smi topo -m

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
--part_config /work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/DGL/DistributedSetup/partitions/ogbn-papers100m_frac_100.0_hops_2_subgraph-1-metis-vol-trans/ogbn-papers100m_frac_100.0_hops_2_subgraph-1-metis-vol-trans.json \
--ip_config ip_config.txt \
"/work/sbajaj_umass_edu/GNNEnv/bin/python3 node_classification.py --dataset ogbn-papers100m --graph_name ogbn-papers100m_frac_100.0_hops_2_subgraph-1-metis-vol-trans --ip_config ip_config.txt --num_epochs 100 --batch_size 1024 --num_gpus 4 --num_hidden 512 --num_layers 2 --fan_out 10,10 --eval_every 2 --lr 0.003"
