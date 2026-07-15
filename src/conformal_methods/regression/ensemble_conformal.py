# General imports
import numpy as np
import pandas as pd

# Sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.base import clone
from sklearn.metrics import r2_score
from mapie.regression import MapieRegressor
from mapie.subsample import Subsample
from src.conformal_methods.regression.split_conformal import SplitConformal
from src.conformal_methods.regression.studentized_conformal import StudentizedConformal
from src.conformal_methods.regression.jackknife_bootstrap import JackknifeBootstrap


class PredCheckedEnsembleJackknifeBootstrap(JackknifeBootstrap):
    def __init__(self, model, num_bootstraps = 500, alpha = 0.1, seed = 42, agg_function = 'mean', top_k = 3, val_set = 'proper', val_size = 0.1):
        self.model = None
        self.init_model = PredCheckedEnsemble(model, top_k = top_k)
        self.models = {model_name: clone(model) for model_name, model in model.items()}
        self.num_bootstraps = num_bootstraps
        self.alpha = alpha
        self.seed = seed
        self.top_k = top_k
        self.cv = None
        self.mapie = None
        self.agg_function = agg_function
        self.val_size = val_size
        self.val_set = val_set
    
    def fit(self, X, y, alpha = None):
        if alpha is None:
            alpha = self.alpha
        self._train(X, y)

    def _train(self, X, y, X_val=None, y_val=None):
        """
        Train the model on the training set.
        """
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.val_size, random_state=self.seed)
        self.init_model.fit(X_train, y_train, X_val, y_val)
        top_k_models = self.init_model.get_top_k_models()
        self.models = {model_name: clone(model) for model_name, model in top_k_models.items()}
        self.model = VotingRegressor(estimators=list(self.models.items()))
        self.cv = Subsample(n_resamplings=self.num_bootstraps, 
                            replace=True, 
                            random_state=self.seed)
        self.mapie = MapieRegressor(estimator=self.model, 
                                    cv=self.cv,
                                    method='plus',
                                    agg_function=self.agg_function,
                                    random_state=self.seed)
        if self.val_set == 'proper':
            self.mapie.fit(X_train, y_train)
        elif self.val_set == 'pcs':
            self.mapie.fit(X, y)


class PredCheckedEnsembleSplitConformal(SplitConformal):
    """
    Split conformal prediction intervals for regression. 

    Args:
        model: sklearn regression model
        alpha: significance level
        seed: random seed for train-test split
    """
    def __init__(self, model, alpha = 0.1, seed = 42, top_k = 3, val_set = 'proper', val_size = 0.5):
        """
        Initialize the SplitConformal class.

        Args:
            model: sklearn regression model
            alpha: significance level
            seed: random seed for train-test split
        """
        self.model = PredCheckedEnsemble(model, top_k = top_k)
        self.models = {model_name: clone(model) for model_name, model in model.items()}
        self.alpha = alpha
        self.seed = seed
        self.q = None 
        self._trained = False
        self._calibrated = False
        self.top_k = top_k
        self.val_set = val_set
        self.val_size = val_size
    def fit(self, X, y, alpha = None):
        """
        Fit and calibrate the model 

        Args:
            X: training features
            y: training labels
        """
        if alpha is None:
            alpha = self.alpha
        X_train, X_calib, y_train, y_calib = train_test_split(X, y, test_size=self.val_size, random_state=self.seed)
        if self.val_set == 'proper':
            self._train(X_train, y_train)
        elif self.val_set == 'pcs':
            self._train(X_train, y_train, X_calib, y_calib)
        self._calibrate(X_calib, y_calib, alpha)
        return self 
    
    def _train(self, X, y, X_val=None, y_val=None):
        """
        Train the model on the training set.
        """
        self.model.fit(X, y, X_val, y_val)
        self._trained = True

class PredCheckedEnsembleStudentizedConformal(StudentizedConformal):
    def __init__(self, mean_model, sd_model, alpha = 0.1, seed = 42, top_k = 3, val_set = 'proper', val_size = 0.5):
        self.mean_model = PredCheckedEnsemble(mean_model, top_k = top_k)
        self.sd_model = sd_model
        self.mean_models = {model_name: clone(model) for model_name, model in mean_model.items()}
        self.sd_models = {model_name: clone(model) for model_name, model in sd_model.items()}
        self.alpha = alpha
        self.seed = seed
        self.q = None 
        self.top_k = top_k
        self.val_set = val_set
        self.val_size = val_size
    def fit(self, X, y, alpha = None):
        """
        Fit and calibrate the model 

        Args:
            X: training features
            y: training labels
        """
        if alpha is None:
            alpha = self.alpha
        X_train, X_calib, y_train, y_calib = train_test_split(X, y, test_size=self.val_size, random_state=self.seed)
        if self.val_set == 'proper':
            self._train(X_train, y_train)
        elif self.val_set == 'pcs':
            self._train(X_train, y_train, X_calib, y_calib)
        self._calibrate(X_calib, y_calib, alpha)

        return self 
    
    def _train(self, X, y, X_val=None, y_val=None):
        """
        Train the model on the training set.
        """
        self.mean_model.fit(X, y, X_val, y_val)
        top_k_models = self.mean_model.top_k_models
        train_residuals = np.abs(y - self.mean_model.predict(X))
        self.sd_model = VotingRegressor(estimators=[(model_name, model) for model_name, model in top_k_models.items()])
        self.sd_model.fit(X, np.abs(train_residuals))

class PredCheckedEnsemble:
    def __init__(self, models, top_k = 3, metric = r2_score):
        self.models = {model_name: clone(model) for model_name, model in models.items()}
        self.voting_regressor = None
        self.pred_scores = {model: -np.inf for model in self.models}
        self.metric = metric
        self.top_k = top_k
        self.top_k_models = None
    
    def fit(self, X, y, X_val=None, y_val=None):
        if (X_val is None) and (y_val is None):
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
        else:
            X_train, y_train = X, y
        self._train(X_train, y_train)
        self._pred_check(X_val, y_val)
        self._get_top_k()
        self._ensemble_models()
        self.voting_regressor.fit(X_train, y_train)
    
    def predict(self, X):
        return self.voting_regressor.predict(X)
    
    def _train(self, X, y):
        for model in self.models:
            self.models[model].fit(X, y)
    
    def _pred_check(self, X, y):
        for model in self.models:
            y_pred = self.models[model].predict(X)
            self.pred_scores[model] = self.metric(y, y_pred)
    
    def _get_top_k(self):
        sorted_models = sorted(self.pred_scores, key=self.pred_scores.get, reverse=True)
        top_k_model_names = sorted_models[:self.top_k]
        self.top_k_models = {model: clone(self.models[model]) for model in top_k_model_names}
    
    def get_top_k_models(self):
        return self.top_k_models

    def _ensemble_models(self):
        self.voting_regressor = VotingRegressor(estimators=list(self.top_k_models.items()))
    
    def clone(self):
        self.models = {model_name: clone(model) for model_name, model in self.models.items()}
        self.top_k_models = None
        self.voting_regressor = None
        return self

if __name__ == "__main__":
    X = np.random.randn(100, 10)
    y = np.random.randn(100)
    models = {'model1': RandomForestRegressor(random_state=42), 
                'model2': RandomForestRegressor(random_state=42), 
                'model3': RandomForestRegressor(random_state=42)}
    conformal = PredCheckedEnsembleSplitConformal(models, top_k=2)
    conformal.fit(X, y)
    print(conformal.predict(X))
    studentized = PredCheckedEnsembleStudentizedConformal(models, models, top_k=2)
    studentized.fit(X, y)
    print(studentized.predict(X))
    jackknife = PredCheckedEnsembleJackknifeBootstrap(models, num_bootstraps=10, top_k=2)
    jackknife.fit(X, y)
    print(jackknife.predict(X))