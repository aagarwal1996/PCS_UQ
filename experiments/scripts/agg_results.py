# Standard imports
import pandas as pd
import numpy as np
import argparse
import os
import pickle
import json
from pathlib import Path

# Experiment configs
from experiments.configs.regression_consts import VALID_UQ_METHODS, VALID_ESTIMATORS, MODELS, DATASETS, SINGLE_CONFORMAL_METHODS
SEEDS = [777, 778, 779, 780, 781, 782, 783, 784, 785, 786]
def agg_results_dataset_method(results_dir, dataset_name = 'data_parkinsons', uq_method = 'split_conformal', estimator = 'XGBoost', train_size = 0.8):
    results = []
    for seed in SEEDS:
        seed_results_dir = Path(f"{results_dir}/{dataset_name}/{uq_method}{estimator}seed_{seed}_train_size_{train_size}_metrics.pkl")
        results.append(pickle.load(open(seed_results_dir, "rb")))
    # Convert list of dictionaries to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate mean and standard error for each metric
    agg_results = {}
    for column in results_df.columns:
        values = results_df[column].values
        agg_results[f"{column}_mean"] = np.mean(values)
        agg_results[f"{column}_std"] = np.std(values) 
    return agg_results

def get_all_uq_methods():
    method_list = []
    for uq_method in VALID_UQ_METHODS:
        for estimator in VALID_ESTIMATORS:
            if uq_method not in SINGLE_CONFORMAL_METHODS:
                method_list.append((f'{uq_method}_', ''))
            elif uq_method == 'LocalConformalRegressor':
                pass
            else:
                method_list.append((f'{uq_method}_', f'{estimator}_'))
    method_list = list(set(method_list))
    return method_list


def main(task = "regression"):
    results_dir = Path(f"experiments/results/{task}/")
    aggregated_results_dir = Path(f'{results_dir}/aggregated_results')
    os.makedirs(aggregated_results_dir, exist_ok=True)
    uq_methods = get_all_uq_methods()
    all_results = {}
    for dataset in DATASETS:
        results_dataset = {}
        for uq_method, estimator in uq_methods:
            results = agg_results_dataset_method(results_dir, dataset, uq_method, estimator)
            # Remove trailing underscores from method and estimator names
            results_dataset[(uq_method.rstrip('_'), estimator.rstrip('_'))] = results
        # Save dataset-specific results
        dataset_results_file = aggregated_results_dir / f"{dataset}_results.pkl"
        with open(dataset_results_file, 'wb') as f:
            pickle.dump(results_dataset, f)
        all_results[dataset] = results_dataset
    # Save all results
    all_results_file = aggregated_results_dir / "all_results.pkl"
    with open(all_results_file, 'wb') as f:
        pickle.dump(all_results, f)
    return all_results
    #print(results)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="regression", help="Task to run")
    args = parser.parse_args()
    main(args.task)

    