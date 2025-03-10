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

def convert_results_to_df(results_dict):    
    # Convert dictionary to DataFrame
    df = pd.DataFrame.from_dict(
        {(dataset, method_hyperparam): metrics
        for dataset, methods in results_dict.items()
        for method_hyperparam, metrics in methods.items()},
        orient='index'
    )

    # # Convert index into MultiIndex
    # df.index = pd.MultiIndex.from_tuples(df.index, names=['Dataset', 'Method', 'Estimator'])

    # # Unstack so that columns follow (Method, Hyperparam) → Metrics format
    # df = df.unstack(level=[1, 2])

    # # Swap levels to achieve desired format (Method (Hyperparam) → Metric 1, Metric 2, ...)
    # df.columns = df.columns.swaplevel(0, 1)
    # df.columns = df.columns.swaplevel(0, 2)
    # df = df.sort_index(axis=1)  # Sort for readability
    return df

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


def main(task = "regression", train_size = 0.8):
    results_dir = Path(f"experiments/results/{task}/")
    aggregated_results_dir = Path(f'{results_dir}/aggregated_results')
    os.makedirs(aggregated_results_dir, exist_ok=True)
    uq_methods = get_all_uq_methods()
    all_results = {}
    for dataset in DATASETS:
        results_dataset = {}
        for uq_method, estimator in uq_methods:
            results = agg_results_dataset_method(results_dir = results_dir, dataset_name = dataset, uq_method = uq_method, estimator = estimator, train_size = train_size)
            # Remove trailing underscores from method and estimator names
            results_dataset[(uq_method.rstrip('_'), estimator.rstrip('_'))] = results
        # Save dataset-specific results
        dataset_results_file = aggregated_results_dir / f"{dataset}_train_size_{train_size}_results.pkl"
        with open(dataset_results_file, 'wb') as f:
            pickle.dump(results_dataset, f)
        all_results[dataset] = results_dataset
    # Save all results
    all_results_file = aggregated_results_dir / f"all_results_train_size_{train_size}.pkl"
    with open(all_results_file, 'wb') as f:
        pickle.dump(all_results, f)
    return all_results
    #print(results)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="regression", help="Task to run")
    parser.add_argument("--train_size", type=float, default=0.8, help="Train size")
    args = parser.parse_args()
    main(args.task, args.train_size)

    