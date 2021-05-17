#!/bin/bash
#SBATCH --account=vigil-hayes
#SBATCH --job-name=get_final_results
#SBATCH --output=/scratch/jfg95/research/tamf/output/get_final_results_%a.txt
#SBATCH --error=/scratch/jfg95/research/tamf/error/get_final_results_%a.txt
#SBATCH --chdir=/scratch/jfg95/research/tamf/
#SBATCH --time=2:00:00
#SBATCH --mem=20000
#SBATCH --array=1-8

# THE PURPOSE OF THIS SCRIPT IS TO GENERATE RESULTS USED IN OUR
# IEEE COMM MAG PAPER SUBMISSION

# will get by-county and state-wide results for U1-U4 and 10%, -1 routes

module load anaconda3
conda activate tamf

# tasks 1-4 will execute by_county code for U1-U4
# tasks 5-8 will execute state-wide for U1-U4

case $SLURM_ARRAY_TASK_ID in
  1)
    x="Utility.NAIVE"
    ;;
  2)
    x="Utility.POPULATED_BLOCKS"
    ;;
  3)
    x="Utility.POPULATED_BLOCKS_W_CONFLICT"
    ;;
  4)
    x="Utility.POPULATED_BLOCKS_W_SCALED_CONFLICT"
    ;;
  5)
    x="Utility.NAIVE"
    ;;
  6)
    x="Utility.POPULATED_BLOCKS"
    ;;
  7)
    x="Utility.POPULATED_BLOCKS_W_CONFLICT"
    ;;
  8)
    x="Utility.POPULATED_BLOCKS_W_SCALED_CONFLICT"
    ;;
esac

if [[ $SLURM_ARRAY_TASK_ID -le 4 ]]
then
    echo "from tamf import *; driver.run_by_county(${x})" | python3
else
    echo "from tamf import *; driver.run_state_wide(${x})" | python3
fi
