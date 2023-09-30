#!/bin/bash

#SBATCH --job-name test   ## name that will show up in the queue
#SBATCH -p gpu-preempt
#SBATCH --gpus=4
#SBATCH --mem=10GB  # memory per CPU core
#SBATCH --time=0-00:01:00  ## time for analysis (day-hour:min:sec)
#SBATCH --constraint=m40

source /work/sbajaj_umass_edu/GNNEnv/bin/activate
DGL/DGL_reference_implementation/fullBatch/sage_scripts/pubmed.sh