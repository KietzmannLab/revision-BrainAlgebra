#!/bin/bash
#SBATCH --job-name=BrainAlgebra
#SBATCH --output=revision_sbatch_output/BrainAlgebra_%j.log
#SBATCH --error=revision_sbatch_output/BrainAlgebra_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=300G
#SBATCH --array=1,3,5,6,7

python3 revision_run_parallel.py ${SLURM_ARRAY_TASK_ID}
