# General Imports
import numpy as np
import pandas as pd
import os
import pickle
# Sklearn Imports
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.ensemble import RandomForestRegressor  
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_error, r2_score

# PCS UQ Imports
from src.metrics.regression_metrics import *
from src.PCS.regression.pcs_uq import PCS_UQ

class PCS_OOB():
    def __init__(self, models, num_bootstraps=10, alpha=0.1, seed=42, top_k=1, save_path=None, load_models=True, metric=r2_score):
        """
        PCS OOB

        Args:
            models: dictionary of model names and models
            num_bootstraps: number of bootstraps
            alpha: significance level
            seed: random seed
            top_k: number of top models to use
            save_path: path to save the models
            load_models: whether to load the models from the save_path
            metric: metric to use for the prediction scores -- assume that higher is better
        """
        self.models = models
        self.num_bootstraps = num_bootstraps
        self.alpha = alpha
        self.seed = seed
        self.top_k = top_k
        self.save_path = save_path
        self.load_models = load_models
        self.metric = metric
        self.pred_scores = {model: -np.inf for model in self.models}
        self.top_k_models = None
        self.bootstrap_models = None
        
    def fit(self, X, y):
        """
        Fit the models
        """
        self._train(X, y)
        self._pred_check(X, y)
        self._get_top_k
        uncalibrated_intervals = self.get_intervals(X)
        

    def _train(self, X, y):
        """
        Train the models and store out-of-bag indices
        """
        # Initialize dictionaries once outside the model loop
        self.bootstrap_models = {}
        self.oob_indices = {}
        
        for model_name, model in self.models.items():
            # Initialize lists for each model
            self.bootstrap_models[model_name] = []
            self.oob_indices[model_name] = []
            
            for i in range(self.num_bootstraps):
                bootstrap_seed = self.seed + i
                # Try to load existing bootstrap model and OOB indices if enabled
                model_path = f"{self.save_path}/pcs_oob/{model_name}_model_seed_{bootstrap_seed}.pkl" if self.save_path else None
                oob_path = f"{self.save_path}/pcs_oob/{model_name}_oob_seed_{bootstrap_seed}.pkl" if self.save_path else None
                bootstrap_model = None
                
                if self.load_models and model_path and os.path.exists(model_path) and os.path.exists(oob_path):
                    with open(model_path, "rb") as f:
                        bootstrap_model = pickle.load(f)
                    with open(oob_path, "rb") as f:
                        oob_indices = pickle.load(f)
                    self.oob_indices[model_name].append(oob_indices)
                else:
                    # Bootstrap the data
                    n_samples = len(X)
                    bootstrap_indices = resample(range(n_samples), n_samples=n_samples, random_state=bootstrap_seed)
                    oob_indices = list(set(range(n_samples)) - set(bootstrap_indices))
                    
                    X_boot = X[bootstrap_indices]
                    y_boot = y[bootstrap_indices]
                    
                    # Store OOB indices
                    self.oob_indices[model_name].append(oob_indices)
                    
                    # Create and fit bootstrap model
                    bootstrap_model = model.__class__(**model.get_params())
                    bootstrap_model.fit(X_boot, y_boot)
                    
                    # Save the bootstrap model and OOB indices if save path is provided
                    if model_path:
                        os.makedirs(os.path.dirname(model_path), exist_ok=True)
                        with open(model_path, "wb") as f:
                            pickle.dump(bootstrap_model, f)
                        with open(oob_path, "wb") as f:
                            pickle.dump(oob_indices, f)
                
                self.bootstrap_models[model_name].append(bootstrap_model)

    def _pred_check(self, X, y):
        """
        Check the predictions of the models using out-of-bag samples.
        Updates self.pred_scores with the average OOB performance of each model.
        """
        for model_name in self.models.keys():
            scores = []
            # For each bootstrap iteration
            for i in range(self.num_bootstraps):
                # Get the OOB indices and corresponding samples
                oob_indices = self.oob_indices[model_name][i]
                X_oob = X[oob_indices]
                y_oob = y[oob_indices]
                
                # Get predictions from the corresponding bootstrap model
                bootstrap_model = self.bootstrap_models[model_name][i]
                y_pred = bootstrap_model.predict(X_oob)
                # Calculate performance metric
                score = self.metric(y_oob, y_pred)
                scores.append(score)
            
            # Update pred_scores with the average OOB score across all bootstraps
            self.pred_scores[model_name] = np.mean(scores)
    
    def _get_top_k(self):
        """
        Get the top k models based on the prediction scores
        """
        # Update top_k_models based on prediction scores
        sorted_models = sorted(self.pred_scores.items(), key=lambda x: x[1], reverse=True)
        self.top_k_models = [model_name for model_name, _ in sorted_models[:self.top_k]]

    def get_intervals(self, X):
        """
        Get the intervals for the top k models
        """
        pass
if __name__ == "__main__":
    models = {
        "rf": RandomForestRegressor(),
        "lr": LinearRegression(),
        "ridge": RidgeCV()
    }
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1)
    X_train, X_calib, y_train, y_calib = train_test_split(X, y, test_size=0.25, random_state=42)
    pcs_oob = PCS_OOB(models, save_path="test", load_models=True)
    pcs_oob.fit(X_train, y_train)
    