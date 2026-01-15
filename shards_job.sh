#!/bin/bash
#SBATCH --account=ai-gwtaylor
#SBATCH --job-name=dnabert2_shards
#SBATCH --mem-per-cpu=16000M      # memory; default unit is megabytes
#SBATCH --time=0-6:00:00
#SBATCH --cpus-per-task=4
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Load modules (Alliance Canada standard)
module load cuda
module load cudnn
module load python/3.11
module load scipy-stack
module load arrow

# Activate virtual environment
source /home/pmillana/projects/aip-gwtaylor/pmillana/venvs/dl-dev/bin/activate


cd /home/pmillana/projects/aip-gwtaylor/pmillana/venvs/dl-dev/CodeRepos/reproduce_dnabert_2

python write_shards.py


