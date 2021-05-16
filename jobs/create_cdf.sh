#!/bin/bash
#SBATCH --account=vigil-hayes
#SBATCH --job-name=create_cdf
#SBATCH --output=/scratch/jfg95/research/tamf/output/create_cdf.txt
#SBATCH --error=/scratch/jfg95/research/tamf/error/create_cdf.txt
#SBATCH --chdir=/scratch/jfg95/research/tamf/
#SBATCH --time=60:00
#SBATCH --mem=10000

module load anaconda3
conda activate tamf

echo "python: $(type python)"
echo "python3: $(type python3)"
python3 ./setup.py install
python3 -u ./scripts/create_cdf.py
