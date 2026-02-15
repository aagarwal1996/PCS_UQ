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
from sklearn.ensemble import RandomForestRegressor  
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.base import clone
from mapie.regression import MapieRegressor
from mapie.subsample import Subsample

# PCS UQ Imports
from src.metrics.regression_metrics import *
from src.PCS.regression.pcs_uq import PCS_UQ

class JackknifeBootstrap(PCS_UQ):
    def __init__(self, model, num_bootstraps=500, alpha=0.1, seed=42, save_path=None, load_models=True, agg_function='mean'):
        """
        Jackknife Bootstrap

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
        self.model = clone(model)
        self.num_bootstraps = num_bootstraps
        self.alpha = alpha
        self.seed = seed
        self.save_path = save_path
        self.load_models = load_models
        self.cv = None
        self.mapie = None
        self.agg_function = agg_function

    def fit(self, X, y, alpha = None):
        """
        Fit the models
        """
        if alpha is None:
            alpha = self.alpha
        self._train(X, y)

        
       
    def _train(self, X, y):
        self.cv = Subsample(n_resamplings=self.num_bootstraps, 
                            replace=True, 
                            random_state=self.seed)

        self.mapie = MapieRegressor(estimator=self.model, 
                                    cv=self.cv,
                                    method='plus',
                                    agg_function=self.agg_function,
                                    random_state=self.seed)
        self.mapie.fit(X, y)
    
    def predict(self, X):
        """
        Make predictions with uncertainty intervals 
        """
        _, y_pis = self.mapie.predict(X, alpha=self.alpha)
        y_pis = np.asarray(y_pis)
        if y_pis.ndim == 3:
            y_pis = y_pis[:, :, 0]
        intervals = np.zeros((X.shape[0], 2))
        lb, ub = y_pis[:, 0], y_pis[:, 1]
        intervals[:, 0] = lb
        intervals[:, 1] = ub
        return intervals


if __name__ == "__main__":
    from time import time
    models = LinearRegression()
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    jackknife_bootstrap = JackknifeBootstrap(models, save_path=None, load_models=False)
    print('Fitting jackknife bootstrap...')
    jackknife_bootstrap.fit(X_train, y_train)
    print('Predicting intervals...')
    intervals = jackknife_bootstrap.predict(X_test)
    print(f'{get_all_metrics(y_test, intervals)}')
