#!/bin/bash
#SBATCH --job-name=BrainAlgebra
#SBATCH --output=BrainAlgebra_%j.log
#SBATCH --error=BrainAlgebra_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=300G

python3 revision_load_data.py