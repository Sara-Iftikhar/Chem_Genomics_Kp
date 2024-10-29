#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH --ntasks-per-node=1
#SBATCH -J xgb_new
#SBATCH --array=1-147
#SBATCH -o junk_outputs/xgb/.%J.out
#SBATCH -e junk_outputs/xgb/.%J.err
#SBATCH --time=20:00:00
#SBATCH --mem=512G

# Specify the path to the config file
config=/ibex/user/iftis0a/klebs_ml/data/config_whole.txt

# Extract the sample name for the current $SLURM_ARRAY_TASK_ID
file_type=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)
target_ID=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)

# Print to a file a message that includes the current $SLURM_ARRAY_TASK_ID, the same name
echo "This is array task ${SLURM_ARRAY_TASK_ID}, the target ID is ${target_ID}, the file type is ${file_type}" >> xgb_output.txt

#run the application:
cd /ibex/user/iftis0a/klebs_ml
python scripts/main.py $SLURM_JOB_ID $file_type $target_ID