# General Imports
import numpy as np
import pandas as pd
import os
import pickle
# Sklearn Imports
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_error, r2_score
class PCS_UQ:
    def __init__(self, models, alpha=0.1, seed=42, top_k = 1, save_path = None, load_models = True, val_size = 0.25, metrics = {'r2': r2_score}):
        """
        PCS UQ

        Args:
            models: dictionary of model names and models
            alpha: significance level
            seed: random seed
            top_k: number of top models to use
            save_path: path to save the models
            load_models: whether to load the models from the save_path
        """
        self.models = models
        self.alpha = alpha
        self.seed = seed
        self.top_k = top_k
        self.save_path = save_path
        self.load_models = load_models
        self.val_size = val_size
        self.metrics = metrics
        self.pred_scores = {metric: {model: -np.inf for model in self.models} for metric in self.metrics}
    
    def fit(self, X, y):
        X_train, X_calib, y_train, y_calib = train_test_split(X, y, test_size=self.val_size, random_state=self.seed)
        self._train(X_train, y_train) # train the models such that they are ready for calibration, saved in self.models
        self._pred_check(X_calib, y_calib) # check the predictions of the models, saved in self.models
        #self._calibrate(X_calib, y_calib) # calibrate the models, saved in self.models

    def _train(self, X, y):
        if self.load_models and (self.save_path is not None):
            print(f"Loading models from {self.save_path}")
            for model in self.models:
                try: 
                    with open(f"{self.save_path}/{model}.pkl", "rb") as f:
                        self.models[model] = pickle.load(f)
                except FileNotFoundError:
                    print(f"No saved model found for {model}")
        else: 
            for model in self.models:
                self.models[model].fit(X, y)
                os.makedirs(self.save_path, exist_ok=True)
                if self.save_path is not None:
                    with open(f"{self.save_path}/{model}.pkl", "wb") as f:
                        pickle.dump(self.models[model], f)

    # For now, assume only one metric. 
    # TODO: Add support for multiple metrics. 
    def _pred_check(self, X, y):
        for model in self.models:
            y_pred = self.models[model].predict(X)
            for metric in self.metrics:
                self.pred_scores[metric][model] = self.metrics[metric](y, y_pred)
    
    def _calibrate(self, X, y):
        pass 

    def predict(self, X):
        pass 

if __name__ == "__main__":
    
    models = {
        "rf": RandomForestRegressor(),
        "lr": LinearRegression()
    }
    X, y = make_regression(n_samples=1000, n_features=10, noise=10)
    pcs_uq = PCS_UQ(models, save_path = 'test', load_models = True)
    pcs_uq.fit(X, y)
    print(pcs_uq.pred_scores)
