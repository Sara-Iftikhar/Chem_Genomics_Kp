#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH --ntasks-per-node=1
#SBATCH -J TD_G4_S
#SBATCH -o junk_outputs/xgb/TD.%J.out
#SBATCH -e junk_outputs/xgb/TD.%J.err
#SBATCH --time=01-00:00:00
#SBATCH --mem=512G

#run the application:

cd /ibex/user/iftis0a/klebs_ml
python scripts/main.py $SLURM_JOB_ID size 4