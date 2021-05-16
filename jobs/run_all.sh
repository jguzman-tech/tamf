#!/bin/bash
#SBATCH --account=vigil-hayes
#SBATCH --job-name=run_all
#SBATCH --output=/scratch/jfg95/research/tamf/output/run_all.txt
#SBATCH --error=/scratch/jfg95/research/tamf/error/run_all.txt
#SBATCH --chdir=/scratch/jfg95/research/tamf/
#SBATCH --time=4:00:00
#SBATCH --mem=20GB

. ./tamf-venv/bin/activate

python3 ./setup.py install
echo "from tamf.driver import *; run_by_county(); run_state_wide()" | python3 -u
