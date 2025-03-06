# General Imports
import numpy as np
import pandas as pd

# Sklearn Imports
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

class PCS_UQ:
    def __init__(self, models, alpha=0.1, seed=42, top_k = 1, save_path = None, load_models = True):
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
    def fit(self, X, y):
        for model in self.models:
            self.models[model].fit(X, y)

        if self.save_path is not None:
            for model in self.models:
                self.models[model].save(self.save_path)

    def pred_check(self, X, y):
        pass 

    def _calibrate(self, X, y):
        pass 

    def predict(self, X):
        pass 

    def _train(self, X, y):
        pass 
    