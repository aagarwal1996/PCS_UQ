# General Import
import numpy as np
import pandas as pd

# Sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator



class LAC:
    def __init__(self, model,alpha = 0.1, cv = 3, random_state = 42, temperature_scaling = True):
        '''
        Implementation of the LAC conformal prediction method for multi-class classification.

        Parameters:
            model: sklearn model (must have a predict_proba method)
            alpha: significance level
            cv: number of folds for cross-validation
            random_state: random state for reproducibility
        '''
        self.model = model
        self.alpha = alpha
        self.cv = cv
        self.random_state = random_state
        self._n_classes = None
        self.temperature_scaling = temperature_scaling
    
    def fit(self, X, y):
        X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=0.5, random_state=self.random_state)
        self.model.fit(X_train, y_train)
        self._n_classes = len(np.unique(y))
        if self.temperature_scaling:
            self._perform_temperature_scaling(X_cal, y_cal)
        
        self.calibrate(X_cal, y_cal)
   
    def _perform_temperature_scaling(self, X_cal, y_cal):
        self.model = FrozenEstimator(self.model)
        self.model = CalibratedClassifierCV(self.model,cv=self.cv)
        self.model.fit(X_cal, y_cal)

    def calibrate(self, X_cal, y_cal):
        class_labels = self.model.classes_
        class_idx = np.argsort(class_labels)
        predicted_probs = self.model.predict_proba(X_cal)[:, class_idx]
        correct_prob_val = predicted_probs[:, y_cal]
        self.q = np.quantile(correct_prob_val, self.alpha)

    def predict(self, X):
        predicted_probs = self.model.predict_proba(X)[:, np.argsort(self.model.classes_)]
        return pd.DataFrame(predicted_probs).apply(lambda r: (r > self.q).astype(int), axis=1, result_type='expand')


if __name__ == "__main__":
    # Example usage of RAPS for classification
    from sklearn.datasets import load_iris
    from sklearn.metrics import accuracy_score
    from src.metrics.classification_metrics import get_coverage, get_mean_width
    
    # Load a sample dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize base model and RAPS
    base_model = RandomForestClassifier(random_state=42)
    lac = LAC(model=base_model, alpha=0.1)
    
    # Fit the model
    lac.fit(X_train, y_train)
    preds = lac.predict(X_test)
    print(preds)
    
    