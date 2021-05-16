#!/bin/bash
#SBATCH --account=vigil-hayes
#SBATCH --job-name=by_county_v2
#SBATCH --output=/scratch/jfg95/research/tamf/output/by_county_v2.txt
#SBATCH --error=/scratch/jfg95/research/tamf/error/by_county_v2.txt
#SBATCH --chdir=/scratch/jfg95/research/tamf/
#SBATCH --time=6:00:00
#SBATCH --mem=20000

module load anaconda3
conda activate tamf

python3 ./setup.py install
python3 -u ./scripts/by_county_v2.py
