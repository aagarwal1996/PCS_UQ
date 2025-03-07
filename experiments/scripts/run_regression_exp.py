
# Standard imports
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from datetime import datetime
import logging
import argparse

# Sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

# PCS imports
from src.PCS.regression.pcs_uq import PCS_UQ
from src.PCS.regression.pcs_oob import PCS_OOB

# Conformal prediction imports
from src.conformal_methods.regression.split_conformal import SplitConformal
from src.conformal_methods.regression.studentized_conformal import StudentizedConformal
from src.conformal_methods.regression.local_conformal import LocalConformalRegressor

# Metrics imports
from src.metrics.regression_metrics import get_all_metrics

# Experiment configs
from experiments.configs.regression_configs import get_regression_datasets, get_conformal_methods, get_pcs_methods
from experiments.configs.regression_consts import VALID_UQ_METHODS, VALID_ESTIMATORS, MODELS, DATASETS, SINGLE_CONFORMAL_METHODS

def run_regression_experiments(
    dataset_name,
    seed,
    uq_method,
    method_name,
    results_dir="experiments/results/regression",
    max_samples=5000, 
    test_size=0.2
):
    X_df, y, bin_df, importance = get_regression_datasets(dataset_name)
    X = X_df.to_numpy()
    X,y = X[:max_samples], y[:max_samples]

    # Create results directory structure
    results_path = Path(results_dir)
    dataset_path = results_path / dataset_name
    seed_path = dataset_path / str(seed)
    
    # Create directories if they don't exist
    seed_path.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    uq_method.fit(X_train, y_train)
    y_pred = uq_method.predict(X_test)
    metrics = get_all_metrics(y_test, y_pred)
    
    # Save metrics as pickle file
    metrics_file = seed_path / f"{method_name}_metrics.pkl"
    with open(metrics_file, 'wb') as f:
        pickle.dump(metrics, f)


# Example usage:
if __name__ == "__main__":
    # Example methods dictionary
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Name of dataset to run experiments on")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--UQ_method", type=str, default="split_conformal", help="UQ method to use")
    parser.add_argument("--estimator", type=str, default="XGBoost", help="Estimator to use")
    args = parser.parse_args()

    # Validate UQ method argument
   

    if args.UQ_method not in VALID_UQ_METHODS:
        raise ValueError(f"Invalid UQ method '{args.UQ_method}'. Must be one of: {VALID_UQ_METHODS}")

    if args.estimator not in VALID_ESTIMATORS:
        raise ValueError(f"Invalid estimator '{args.estimator}'. Must be one of: {VALID_ESTIMATORS}")
    
    if args.UQ_method in SINGLE_CONFORMAL_METHODS:
        uq_method, method_name  = get_conformal_methods(args.UQ_method, args.estimator)
    
    elif args.UQ_method == "majority_vote":
        uq_method, method_name = get_conformal_methods("majority_vote", args.estimator)
        method_name = f"majority_vote"
    
    elif args.UQ_method == "pcs_uq":
        uq_method  = get_pcs_methods("pcs_uq")
        method_name = "pcs_uq"
    
    elif args.UQ_method == "pcs_oob":
        uq_method  = get_pcs_methods("pcs_oob")
        method_name = "pcs_oob"
    
    # Set random seed
    np.random.seed(args.seed)

    run_regression_experiments(args.dataset, args.seed, uq_method, method_name)