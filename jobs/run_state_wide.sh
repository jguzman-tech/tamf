#!/bin/bash
#SBATCH --account=vigil-hayes
#SBATCH --job-name=run_state_wide
#SBATCH --output=/scratch/jfg95/research/tamf/output/run_state_wide.txt
#SBATCH --error=/scratch/jfg95/research/tamf/error/run_state_wide.txt
#SBATCH --chdir=/scratch/jfg95/research/tamf/
#SBATCH --time=6:00:00
#SBATCH --mem=20GB

. ./tamf-compute-venv/bin/activate

echo "about to enter python..."
python3 ./scripts/run_state_wide.py
