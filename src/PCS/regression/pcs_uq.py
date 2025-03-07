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
from sklearn.base import clone
# PCS UQ Imports
from src.metrics.regression_metrics import *

class PCS_UQ:
    def __init__(self, models, num_bootstraps=10, alpha=0.1, seed=42, top_k = 1, save_path = None, load_models = True, val_size = 0.25, metric = r2_score):
        """
        PCS UQ

        Args:
            models: dictionary of model names and models
            alpha: significance level
            seed: random seed
            top_k: number of top models to use
            save_path: path to save the models
            load_models: whether to load the models from the save_path
            metric: metric to use for the prediction scores -- assume that higher is better
        """
        self.models = {model_name: clone(model) for model_name, model in models.items()}
        self.alpha = alpha
        self.num_bootstraps = num_bootstraps
        self.seed = seed
        self.top_k = top_k
        self.save_path = save_path
        self.load_models = load_models
        self.val_size = val_size
        self.metric = metric
        self.pred_scores = {model: -np.inf for model in self.models}
        self.top_k_models = None
        self.bootstrap_models = None
    
    def fit(self, X, y, alpha = 0.1):
        """
        Args: 
            X: features
            y: target
        Returns: 
            None
        Steps: 
        1. Split the data into training and calibration sets
        2. Train the models
        3. Check the predictions of the models
        4. Get the top k models
        5. Calibrate the top-k models 
        """
        if alpha is None:
            alpha = self.alpha
        X_train, X_calib, y_train, y_calib = train_test_split(X, y, test_size=self.val_size, random_state=self.seed)
        self._train(X_train, y_train) # train the models such that they are ready for calibration, saved in self.models
        self._pred_check(X_calib, y_calib) # check the predictions of the models, saved in self.models
        self.top_k_models = self._get_top_k()
        self._fit_bootstraps(X_train, y_train)
        uncalibrated_intervals  = self.get_intervals(X_calib) # get the uncalibrated intervals and raw width/coverage
        self.uncalibrated_metrics = get_all_metrics(y_calib, uncalibrated_intervals[:,[0,2]]) # drop median to assess raw coverage and width
        gamma = self.calibrate(uncalibrated_intervals, y_calib) # calibrate the intervals to get the best gamma 

                

    def _train(self, X, y):
        if self.load_models and (self.save_path is not None):
            print(f"Loading models from {self.save_path}")
            for model in self.models:
                try: 
                    with open(f"{self.save_path}/pcs_uq/{model}.pkl", "rb") as f:
                        self.models[model] = pickle.load(f)
                except FileNotFoundError:
                    print(f"No saved model found for {model}, fitting new model")
                    self.models[model].fit(X, y)
                    os.makedirs(f"{self.save_path}/pcs_uq", exist_ok=True)
                    with open(f"{self.save_path}/pcs_uq/{model}.pkl", "wb") as f:
                        pickle.dump(self.models[model], f)
        else: 
            for model in self.models:
                self.models[model].fit(X, y)
                if self.save_path is not None:
                    os.makedirs(f"{self.save_path}/pcs_uq", exist_ok=True)
                    with open(f"{self.save_path}/pcs_uq/{model}.pkl", "wb") as f:
                        pickle.dump(self.models[model], f)

    # For now, assume only one metric. 
    # TODO: Add support for multiple metrics by picking in average highest rank
    def _pred_check(self, X, y):
        """
        Args: 
            X: features
            y: target
        Steps: 
        1. Predict the target using the models
        2. Calculate the prediction score for each model
        """
        for model in self.models:
            y_pred = self.models[model].predict(X)
            self.pred_scores[model] = self.metric(y, y_pred)

    def _get_top_k(self):
        """
        Args: 
            None
        Steps: 
        1. Sort the models by the prediction score
        2. Return the top k models
        """
        sorted_models = sorted(self.pred_scores, key=self.pred_scores.get, reverse=True)
        top_k_model_names = sorted_models[:self.top_k]
        self.top_k_models = {model: self.models[model] for model in top_k_model_names}
        return self.top_k_models
    
    def _fit_bootstraps(self, X, y):
        """
        Generate prediction intervals using bootstrap resampling for each top-k model
        
        Args:
            X: features
            y: target
        Returns:
            Dictionary of model predictions for each bootstrap sample
        """
        bootstrap_predictions = {model: [] for model in self.top_k_models}
        bootstrap_models = {model: [] for model in self.top_k_models}
        
        for i in range(self.num_bootstraps):
            for model_name, model in self.top_k_models.items():
                if self.load_models and self.save_path is not None:
                    # Try to load existing bootstrap model
                    bootstrap_dir = os.path.join(self.save_path, 'pcs_uq', 'bootstrap_models', model_name)
                    bootstrap_path = f"{bootstrap_dir}/bootstrap_{model_name}_{i}.pkl"
                    
                    try:
                        with open(bootstrap_path, "rb") as f:
                            bootstrap_model = pickle.load(f)
                            print(f"Loaded bootstrap model {i} for {model_name}")
                    except (FileNotFoundError, EOFError):
                        # If loading fails, fit a new bootstrap model
                        print(f"Fitting new bootstrap model {i} for {model_name}")
                        X_boot, y_boot = resample(X, y, random_state=self.seed + i)
                        bootstrap_model = model.__class__(**model.get_params())
                        bootstrap_model.fit(X_boot, y_boot)
                        
                        # Save the newly fitted model
                        if self.save_path is not None:
                            os.makedirs(bootstrap_dir, exist_ok=True)
                            with open(bootstrap_path, "wb") as f:
                                pickle.dump(bootstrap_model, f)
                else:
                    # Fit new bootstrap model without attempting to load
                    X_boot, y_boot = resample(X, y, random_state=self.seed + i)
                    bootstrap_model = model.__class__(**model.get_params())
                    bootstrap_model.fit(X_boot, y_boot)
                    
                    # Save the model if save_path is specified
                    if self.save_path is not None:
                        bootstrap_dir = os.path.join(self.save_path, 'pcs_uq', 'bootstrap_models', model_name)
                        os.makedirs(bootstrap_dir, exist_ok=True)
                        with open(f"{bootstrap_dir}/bootstrap_{model_name}_{i}.pkl", "wb") as f:
                            pickle.dump(bootstrap_model, f)
                
                # Store the bootstrap model
                bootstrap_models[model_name].append(bootstrap_model)
                
                # Get predictions for the original data
                predictions = bootstrap_model.predict(X)
                bootstrap_predictions[model_name].append(predictions)
        
        self.bootstrap_models = bootstrap_models
        #return bootstrap_predictions


    def get_intervals(self, X):
        """
        Args: 
            X: features
            gamma: expansion factor
        Returns: 
            Dictionary containing lower and upper bounds of prediction intervals combining all models
            Format: {'lower': array, 'upper': array, 'predictions': array}
        """
        # Get number of samples
        n_samples = X.shape[0]
        
        # Initialize array to store all predictions
        # Shape will be (n_samples, K*B) - each row represents all predictions for one data point
        all_predictions = np.zeros((n_samples, 0))
        
        # Collect predictions for each data point
        for model_name, bootstrap_models in self.bootstrap_models.items():
            for model in bootstrap_models:
                pred = model.predict(X).reshape(-1, 1)  # Shape: (n_samples, 1)
                all_predictions = np.hstack((all_predictions, pred))
        
        # Sort predictions for each data point
        sorted_predictions = np.sort(all_predictions, axis=1)  # Sort along K*B axis
        intervals = np.array((n_samples, 3))
        # Calculate quantiles for each row
        intervals = np.zeros((n_samples, 3))
        intervals[:, 0] = np.quantile(sorted_predictions, self.alpha/2, axis=1)  # Lower bound
        intervals[:, 1] = np.quantile(sorted_predictions, 0.5, axis=1)  # Median
        intervals[:, 2] = np.quantile(sorted_predictions, 1.0 - self.alpha/2, axis=1)  # Upper bound
        return intervals

        
       
    def calibrate(self, intervals, y_calib):
        """
        Calibrate the intervals
        """
        gamma_min = 0.0
        gamma_max = 100 # TODO: find a better upper bound
        gamma_range = np.linspace(start = gamma_min, stop = gamma_max, num = int(1e5))
        coverage_list = []
        width_list = []
        # TODO: binary search for the best gamma
        best_gamma = 0.0
        for gamma in gamma_range:
            lb = intervals[:, 0] - gamma * (intervals[:, 1] - intervals[:, 0])
            ub = intervals[:, 1] + gamma * (intervals[:, 2] - intervals[:, 1])
            coverage = np.mean((y_calib >= lb) & (y_calib <= ub))
            if coverage >= 1.0 - self.alpha:
                best_gamma = gamma
                break
        self.gamma = best_gamma
        return best_gamma

    def predict(self, X):
        uncalibrated_intervals = self.get_intervals(X)
        lower_bound = uncalibrated_intervals[:, 0] - self.gamma * (uncalibrated_intervals[:, 1] - uncalibrated_intervals[:, 0])
        upper_bound  = uncalibrated_intervals[:, 1] + self.gamma * (uncalibrated_intervals[:, 2] - uncalibrated_intervals[:, 1])
        intervals = np.zeros((X.shape[0], 2))
        intervals[:, 0] = lower_bound
        intervals[:, 1] = upper_bound
        return intervals



if __name__ == "__main__":
    
    models = {
        "rf": RandomForestRegressor(),
        "lr": LinearRegression(), 
        'ridge': RidgeCV()
    }
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    pcs_uq = PCS_UQ(models, save_path = 'test', load_models = True)
    pcs_uq.fit(X_train, y_train)
    intervals = pcs_uq.predict(X_test)
    print(get_all_metrics(y_test, intervals))

 # # Calculate indices for lower and upper bounds based on alpha
        # n_total = sorted_predictions.shape[1]  # Total number of predictions per point (K * B)
        # lower_idx = int(np.floor(self.alpha/2 * n_total))
        # upper_idx = int(np.ceil((1 - self.alpha/2) * n_total))
        
        # # Get the prediction intervals
        # intervals = {
        #     'lower': sorted_predictions[:, lower_idx],
        #     'upper': sorted_predictions[:, upper_idx],
        #     'predictions': sorted_predictions  # Including all predictions for potential further use
        # }
        
        # return intervals
