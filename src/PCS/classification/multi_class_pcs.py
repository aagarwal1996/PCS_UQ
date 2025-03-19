# General Imports
import numpy as np
import pandas as pd
import os
import pickle
import copy
from tqdm import tqdm
# Sklearn Imports
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier  
from sklearn.linear_model import RidgeClassifierCV, LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score, log_loss
from sklearn.base import clone


# PCS UQ Imports
from calibration_utils import model_prop_calibration, predict_model_prop_calibration, APS_calibration, predict_APS_calibration
from src.metrics.classification_metrics import get_all_metrics

class MultiClassPCS:
    def __init__(self, models, num_bootstraps=100, alpha=0.1, seed=42, top_k = 1, save_path = None, load_models = True, val_size = 0.25, 
                 metric = log_loss, calibration_method = 'model_prop'):
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
        self.models = {model_name: copy.deepcopy(model) for model_name, model in models.items()}
        self.alpha = alpha
        self.num_bootstraps = num_bootstraps
        self.seed = seed
        self.top_k = top_k
        self.save_path = save_path
        self.load_models = load_models
        self.val_size = val_size
        self.metric = metric
        self.pred_scores = {model: np.inf for model in self.models}
        self.top_k_models = None
        self.bootstrap_models = None
        self.calibration_method = calibration_method
        self.n_classes = None
    
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
        self.n_classes = len(np.unique(y))
        if alpha is None:
            alpha = self.alpha
        X_train, X_calib, y_train, y_calib = train_test_split(X, y, test_size=self.val_size, random_state=self.seed)
        self._train(X_train, y_train) # train the models such that they are ready for calibration, saved in self.models
        self._pred_check(X_calib, y_calib) # check the predictions of the models, saved in self.models
        self.top_k_models = self._get_top_k()
        self._fit_bootstraps(X_train, y_train)
        #uncalibrated_intervals  = self.get_intervals(X_calib) # get the uncalibrated intervals and raw width/coverage
        self.gamma, self.temperature = self.calibrate(X_calib, y_calib) # calibrate the intervals to get the best gamma 
        self.prediction_sets = self.predict(X_test)

                

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
            y_pred = self.models[model].predict_proba(X)
            self.pred_scores[model] = self.metric(y, y_pred)
    def _get_top_k(self):
        """
        Args: 
            None
        Steps: 
        1. Sort the models by the prediction score
        2. Return the top k models
        """
        sorted_models = sorted(self.pred_scores, key=self.pred_scores.get)
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
        
        for i in tqdm(range(self.num_bootstraps)):
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
                        bootstrap_model = copy.deepcopy(model)
                        bootstrap_model.fit(X_boot, y_boot)
                        
                        # Save the newly fitted model
                        if self.save_path is not None:
                            os.makedirs(bootstrap_dir, exist_ok=True)
                            with open(bootstrap_path, "wb") as f:
                                pickle.dump(bootstrap_model, f)
                else:
                    # Fit new bootstrap model without attempting to load
                    X_boot, y_boot = resample(X, y, random_state=self.seed + i)
                    bootstrap_model = copy.deepcopy(model)
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

    def calibrate(self, X,y):
        if self.calibration_method == 'model_prop':
            gamma, temperature = model_prop_calibration(X, y, self.bootstrap_models, self.alpha)
            return gamma, temperature
        elif self.calibration_method == 'APS':
            gamma, temperature = APS_calibration(X, y, self.bootstrap_models, self.alpha)
            return gamma, temperature
        else:
            raise ValueError(f"Calibration method {self.calibration_method} not supported")
        
    def predict(self, X):
        if self.calibration_method == 'model_prop':
            return predict_model_prop_calibration(X, self.bootstrap_models, self.gamma, self.n_classes, self.temperature)
        elif self.calibration_method == 'APS':
            return predict_APS_calibration(X, self.bootstrap_models, self.gamma, self.n_classes, self.temperature)
        else:
            raise ValueError(f"Calibration method {self.calibration_method} not supported")
    
    
        

if __name__ == "__main__":
    
    models = {
        "rf": RandomForestClassifier(n_estimators=5, min_samples_leaf=50, random_state=42),
        'logistic': LogisticRegression(random_state=42)
    }
    X, y = make_classification(n_samples=250, n_features=10, n_classes=10, n_informative=5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    calibration_method = ['APS']
    for method in calibration_method:
        pcs_uq = MultiClassPCS(models, save_path = 'test', load_models = False, calibration_method = method)
        pcs_uq.fit(X_train, y_train)
        predictions_sets =pcs_uq.predict(X_test)
        metrics = get_all_metrics(y_test, predictions_sets)
        print(metrics)

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
