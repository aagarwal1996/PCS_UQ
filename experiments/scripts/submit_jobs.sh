#!/bin/bash
#SBATCH --job-name=regression_exp
#SBATCH --output=logs/regression_exp_%A_%a.out
#SBATCH --error=logs/regression_exp_%A_%a.err
#SBATCH --time=8:00:00  # Adjust time as needed
#SBATCH --cpus-per-task=1  # Each task gets 1 CPU
#SBATCH --mem=16G  # Adjust memory per job
#SBATCH --ntasks=1
#SBATCH --partition=jsteinhardt
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aa3797@berkeley.edu
#SBATCH --array=0-6000

# Activate Conda environment
module load python/3.10  # Change this based on your cluster setup
conda init
conda activate pcs_uq

# Define parameters
DATASETS=("data_parkinsons" "data_airfoil" "data_computer" "data_concrete" "data_powerplant"
          "data_miami_housing" "data_insurance" "data_qsar" "data_allstate" "data_mercedes"
          "data_energy_efficiency" "data_kin8nm" "data_naval_propulsion" "data_diamond"
          "data_superconductor" "data_ca_housing" "data_protein_structure")

UQ_METHODS=("split_conformal" "studentized_conformal" "majority_vote"
            "LocalConformalRegressor" "pcs_uq" "pcs_oob")

ALL_ESTIMATORS=("XGBoost" "RandomForest" "ExtraTrees" "AdaBoost"
                "OLS" "Ridge" "Lasso" "ElasticNet" "MLP")

REDUCED_ESTIMATORS=("XGBoost")  # For majority_vote, pcs_uq, pcs_oob

SEEDS=(0 1 2 3 4 5 6 7 8 9)  # Modify as needed

# Calculate total job count
TOTAL_JOBS=0
for uq in "${UQ_METHODS[@]}"; do
    if [[ "$uq" == "majority_vote" || "$uq" == "pcs_uq" || "$uq" == "pcs_oob" ]]; then
        estimators=("${REDUCED_ESTIMATORS[@]}")
    else
        estimators=("${ALL_ESTIMATORS[@]}")
    fi
    TOTAL_JOBS=$(( TOTAL_JOBS + ${#DATASETS[@]} * ${#SEEDS[@]} * ${#estimators[@]} ))
done

# **Debugging Statement**: Print Total Jobs
echo "Total Jobs: $TOTAL_JOBS"

# Ensure we have at least one job (avoid invalid array error)
if [[ "$TOTAL_JOBS" -le 0 ]]; then
    echo "Error: No jobs to submit. Check dataset, UQ method, or estimator definitions."
    exit 1
fi


# Compute task index
TASK_ID=$SLURM_ARRAY_TASK_ID

# Compute dataset index
dataset_idx=0
uq_idx=0
estimator_idx=0
seed_idx=0
job_counter=0

for uq in "${UQ_METHODS[@]}"; do
    if [[ "$uq" == "majority_vote" || "$uq" == "pcs_uq" || "$uq" == "pcs_oob" ]]; then
        estimators=("${REDUCED_ESTIMATORS[@]}")
    else
        estimators=("${ALL_ESTIMATORS[@]}")
    fi

    for dataset in "${DATASETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            for estimator in "${estimators[@]}"; do
                if [[ "$job_counter" -eq "$TASK_ID" ]]; then
                    DATASET="$dataset"
                    UQ_METHOD="$uq"
                    ESTIMATOR="$estimator"
                    SEED="$seed"
                fi
                job_counter=$((job_counter + 1))
            done
        done
    done
done

echo "Running job: $DATASET $UQ_METHOD $ESTIMATOR $SEED"
# Run the Python script
python experiments/scripts/run_regression_exp.py --dataset "$DATASET" --UQ_method "$UQ_METHOD" --seed "$SEED" --estimator "$ESTIMATOR"
