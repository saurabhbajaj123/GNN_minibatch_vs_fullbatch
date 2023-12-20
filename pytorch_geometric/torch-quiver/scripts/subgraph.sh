#!/bin/bash

#SBATCH --job-name papers100m  ## name that will show up in the queue
#SBATCH --mem=500GB  # memory per CPU core
#SBATCH --partition=cpu
#SBATCH --time=0-24:00:00  ## time for analysis (day-hour:min:sec)

source /work/sbajaj_umass_edu/pygenv1/bin/activate

echo y | python preprocess/subgraph.py