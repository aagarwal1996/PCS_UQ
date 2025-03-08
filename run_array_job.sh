#!/bin/bash
#SBATCH --job-name=regression_exp
#SBATCH --output=logs/regression_%A_%a.out
#SBATCH --error=logs/regression_%A_%a.err
#SBATCH --array=1-4590
#SBATCH --time=8:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Create logs directory if it doesn't exist
mkdir -p logs

# Get the combination for this array job
combination=$(sed -n "${SLURM_ARRAY_TASK_ID}p" job_combinations.txt)
dataset=$(echo $combination | cut -d' ' -f1)
uq_method=$(echo $combination | cut -d' ' -f2)
estimator=$(echo $combination | cut -d' ' -f3)
seed=$(echo $combination | cut -d' ' -f4)

# Run the experiment
if [ "$estimator" = "none" ]; then
    python run_regression_exp.py --dataset $dataset --uq_method $uq_method --seed $seed
else
    python run_regression_exp.py --dataset $dataset --uq_method $uq_method --estimator $estimator --seed $seed
fi
