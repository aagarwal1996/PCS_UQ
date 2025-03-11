
    # General imports
import numpy as np
import pandas as pd

# Sklearn imports
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
from sklearn.datasets import make_classification
# Local imports
from src.PCS.classification.calibration_utils import temperature_scaling
from src.metrics.classification_metrics import get_all_metrics

# Mapie imports
from mapie.classification import MapieClassifier
from mapie.conformity_scores import RAPSConformityScore, LACConformityScore, NaiveConformityScore, APSConformityScore 

class MultiClassConformal:
   
    def __init__(self, model, alpha = 0.1, seed = 42, conformity_score = 'RAPS', temperature_scaling = True):
        """
        Initialize the RAPSConformal class.

        Args:
            model: sklearn classification model
            alpha: significance level
            seed: random seed for train-test split
        """
        self.model = clone(model)
        self.alpha = alpha
        self.seed = seed
        self.q = None 
        self._trained = False
        self._calibrated = False
        self.temperature_scaling = temperature_scaling
        if conformity_score == 'RAPS':
            self.conf_score = RAPSConformityScore()
        elif conformity_score == 'LAC':
            self.conf_score = LACConformityScore()
        elif conformity_score == 'Naive':
            self.conf_score = NaiveConformityScore()
        elif conformity_score == 'APS':
            self.conf_score = APSConformityScore()
        else:
            raise ValueError("Conformity score must be one of 'RAPS', 'LAC', 'Naive', or 'APS'")
        self.mapie_classifier = None
    
    def fit(self, X, y, alpha = None):
        """
        Fit and calibrate the model 

        Args:
            X: training features
            y: training labels
        """
        if alpha is None:
            alpha = self.alpha
        X_train, X_calib, y_train, y_calib = train_test_split(X, y, test_size=0.5, random_state=self.seed)    
        self._train(X_train, y_train)
        self._temperature_scaling(X_calib, y_calib) # Perform temperature scaling on the calibration set
        self.mapie_classifier = MapieClassifier(estimator=self.model, 
                                                cv='prefit', 
                                                conformity_score=self.conf_score,
                                                random_state=self.seed)
        self._calibrate(X_calib, y_calib) # Calibrate the model on the calibration set

        return self 
    
    

    def _train(self, X, y):
        """
        Train the model on the training set.
        """
        
        self.model.fit(X, y)
        self._trained = True
    
    def _temperature_scaling(self, X, y):
        """
        Temperature scaling the model on the training set.
        """
        if self.temperature_scaling:
            self.model = CalibratedClassifierCV(self.model, method='sigmoid', cv = 'prefit')
            self.model.fit(X, y)
        else:
            self.model = self.model

    def _calibrate(self, X, y):
        """
        Calibrate the model on the validation set.
        """
        if self._trained == False:
            raise ValueError("Model must be fitted before calling calibrate.")
        self.mapie_classifier.fit(X, y)
        

    def predict(self, X, alpha=None):
        """
        Predicts intervals for test data using calibration errors.

        Parameters:
        ----------
        X : np.ndarray
            
        Returns:
        -------
        np.ndarray
            An array of shape (n_samples, n_class) containing the lower and upper bounds of the prediction intervals.
        """
        if alpha is None:
            alpha = self.alpha
        # Predict on the test set
        _ , y_pred_test = self.mapie_classifier.predict(X, alpha, 
                                                        include_last_label='randomized')

        return y_pred_test[:, :, 0].astype(int)


if __name__ == "__main__":
    X, y = make_classification(n_samples=2000, n_features=10, n_classes=3, n_informative=5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = RandomForestClassifier()
    conformity_scores = ['RAPS', 'LAC', 'Naive', 'APS']
    for conformity_score in conformity_scores:
        split_conformal = MultiClassConformal(model, conformity_score=conformity_score, temperature_scaling=True)
        split_conformal.fit(X, y)
        print(f'Conformity score: {conformity_score}, Metrics: {get_all_metrics(y, split_conformal.predict(X))}')