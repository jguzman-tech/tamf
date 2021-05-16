#!/bin/bash
#SBATCH --account=vigil-hayes
#SBATCH --job-name=by_county
#SBATCH --output=/scratch/jfg95/research/tamf/output/by_county.txt
#SBATCH --error=/scratch/jfg95/research/tamf/error/by_county.txt
#SBATCH --chdir=/scratch/jfg95/research/tamf/
#SBATCH --time=6:00:00
#SBATCH --mem=20000

module load anaconda3
conda activate tamf

echo "python: $(type python)"
echo "python3: $(type python3)"
python3 ./setup.py install
python3 -u ./scripts/by_county.py
