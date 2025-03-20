import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.base import clone
from src.conformal_methods.classification.multi_class_conformal import MultiClassConformal
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from src.metrics.classification_metrics import get_all_metrics

class MajorityVoteClassifier:
    def __init__ (self, models, alpha = 0.1, seed = 42, conformity_score = 'APS', temperature_scaling = True):
        """
        Initialize the Conformal Majority Vote model

        Args:
            models: dictionary of model names and models
            alpha: significance level
            seed: random seed
        """
        self.models = {model_name: clone(model) for model_name, model in models.items()}
        self.conformals = {model_name: None for model_name in models.keys()}
        self.K = len(models)
        self.alpha = alpha
        self.seed = seed
        self.conformity_score = conformity_score
        self.temperature_scaling = temperature_scaling
        self.trained = False


    def fit(self, X, y, alpha = None):
        """
        Fit the model on the training data.

        Args:
            X: training features
            y: training labels
        """
        if alpha is None:
            alpha = self.alpha
        self.n_classes = len(np.unique(y))
        self.conformals = {}
        for model_name, model in self.models.items():
            conformal = MultiClassConformal(model, self.alpha/2, self.seed, self.conformity_score, self.temperature_scaling)
            conformal.fit(X, y)
            self.conformals[model_name] = conformal
        self.trained = True
        return self
        
    
    def predict(self, X, tau = 0.5):
        """
        Predict the intervals for the test set.

        Args:
            X: test features
            tau: significance level
        """
        if self.trained == False:
            raise ValueError("Model must be trained before calling predict.")
        prediction_sets = np.zeros((X.shape[0], self.n_classes, self.K))
        count = 0
        for model_name, conformal in self.conformals.items():
            prediction_sets[:, :, count] = conformal.predict(X)
            count += 1
        averaged_prediction_sets = np.mean(prediction_sets, axis=2)
        # Threshold the averaged prediction sets at tau
        thresholded_prediction_sets = (averaged_prediction_sets >= tau).astype(int)
        return thresholded_prediction_sets
        
    
if __name__ == "__main__":
    X,y = make_classification(n_samples=1000, n_features=10, n_classes=3, n_informative=5, n_redundant=0, n_clusters_per_class=1)
    models = {
        'rf': RandomForestClassifier(),
        'lr': LogisticRegression(),
        'extraTrees': ExtraTreesClassifier()
    }
    conformal_majority_vote = MajorityVoteClassifier(models)
    conformal_majority_vote.fit(X, y)
    print(f' Metrics: {get_all_metrics(y, conformal_majority_vote.predict(X))}')

    