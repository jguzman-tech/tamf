#!/bin/bash
#SBATCH --account=vigil-hayes
#SBATCH --job-name=test
#SBATCH --output=/scratch/jfg95/research/tamf/output/test.txt
#SBATCH --error=/scratch/jfg95/research/tamf/error/test.txt
#SBATCH --chdir=/scratch/jfg95/research/tamf/
#SBATCH --time=2:00
#SBATCH --mem=20GB

# . ./tamf-venv/bin/activate
source ./tamf-venv/bin/activate
type python3
pwd
time echo "from tamf.driver import *; test()" | python3
