
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

def run_uq_experiments(
    methods,
    X,
    y,
    dataset_name,
    split_name,
    test_size=0.2,
    random_state=42,
    results_dir="results"
):
    """
    Run uncertainty quantification experiments for multiple methods.
    
    Args:
        methods (dict): Dictionary of UQ method names and their instances
        X (array-like): Features
        y (array-like): Targets
        dataset_name (str): Name of the dataset
        split_name (str): Name of the random split
        test_size (float): Proportion of dataset to use for testing
        random_state (int): Random seed for reproducibility
        results_dir (str): Directory to save results
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(f"Data split: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples")
    
    # Create results directory
    results_path = Path(results_dir) / dataset_name / split_name
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Store results for all methods
    all_results = {}
    
    for method_name, method in methods.items():
        logger.info(f"Running {method_name}...")
        
        try:
            # Fit the method
            method.fit(X_train, y_train)
            
            # Get predictions and uncertainties
            y_pred, uncertainties = method.predict(X_test)
            
            # Calculate basic metrics
            mse = np.mean((y_test - y_pred) ** 2)
            mae = np.mean(np.abs(y_test - y_pred))
            
            # Store results
            method_results = {
                "predictions": y_pred.tolist(),
                "uncertainties": uncertainties.tolist(),
                "metrics": {
                    "mse": float(mse),
                    "mae": float(mae),
                },
                "timestamp": datetime.now().isoformat()
            }
            
            all_results[method_name] = method_results
            
            # Save individual method results
            method_file = results_path / f"{method_name}_results.json"
            with open(method_file, "w") as f:
                json.dump(method_results, f, indent=2)
                
            logger.info(f"Results saved for {method_name}")
            
        except Exception as e:
            logger.error(f"Error running {method_name}: {str(e)}")
            continue
    
    # Save summary results
    summary_file = results_path / "summary.json"
    summary = {
        "dataset": dataset_name,
        "split": split_name,
        "methods": {
            name: results["metrics"]
            for name, results in all_results.items()
        }
    }
    
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info("Experiment completed successfully!")
    return all_results

# Example usage:
if __name__ == "__main__":
    # Example methods dictionary
    methods = {
        "method1": UQMethod1(),
        "method2": UQMethod2(),
    }
    
    # Run experiments
    results = run_uq_experiments(
        methods=methods,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        dataset_name="example_dataset",
        split_name="split_1"
    )
