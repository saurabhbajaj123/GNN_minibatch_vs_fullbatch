#!/bin/bash

#SBATCH --job-name products   ## name that will show up in the queue
#SBATCH --mem=100G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --partition=cpu-preempt
#SBATCH --time=02:00:00          # total run time limit (HH:MM:SS)


source /work/sbajaj_umass_edu/GNNEnv/bin/activate


python cross_gpu_data_transfer.py