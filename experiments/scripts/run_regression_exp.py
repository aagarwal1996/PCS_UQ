
# Standard imports
import numpy as np
import pandas as pd
from pathlib import Path
import json
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
from experiments.configs.regression_configs import get_regression_datasets, MODELS, get_conformal_methods, get_pcs_methods, get_uq_methods

def run_regression_experiments(
    dataset_name,
    seed,
    results_dir="experiments/results/regression"
):
    X_df, y, bin_df, importance = get_regression_datasets(dataset_name)
    X = X_df.to_numpy()
    UQ_models = get_uq_methods(MODELS)
    X,y = X[:5000], y[:5000]

    # Create results directory structure
    results_path = Path(results_dir)
    dataset_path = results_path / dataset_name
    seed_path = dataset_path / str(seed)
    
    # Create directories if they don't exist
    seed_path.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # Run experiments
    for model_name, model in UQ_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = get_all_metrics(y_test, y_pred)
        print(f"{model_name}: {metrics}")

# Example usage:
if __name__ == "__main__":
    # Example methods dictionary
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Name of dataset to run experiments on")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    run_regression_experiments(args.dataset, args.seed)