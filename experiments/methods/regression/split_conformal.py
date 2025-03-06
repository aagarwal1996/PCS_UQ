import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

class SplitConformal:
    """
    Split conformal prediction intervals for regression. 

    Args:
        model: sklearn regression model
        alpha: significance level
        seed: random seed for train-test split
    """
    def __init__(self, model, alpha = 0.1, seed = 42):

        self.model = model
        self.alpha = alpha
        self.seed = seed
        self.q = None 
    
    def fit(self, X, y):
        """
        Fit and calibrate the model 

        Args:
            X: training features
            y: training labels
        """
        X_train, X_calib, y_train, y_calib = train_test_split(X, y, test_size=0.5, random_state=self.seed)    
        self.model.fit(X_train, y_train)
        self._calibrate(X_calib, y_calib)

        return self 

    def _calibrate(self, X, y):
        """
        Calibrate the model on the validation set.
        """
        residuals = np.abs(y - self.model.predict(X))
        self.q = np.sort(residuals)[int(np.ceil((len(X) + 1) * (1 - self.alpha)) - 1)]


    def predict(self, X):
        """
        Predicts intervals for test data using calibration errors.

        Parameters:
        ----------
        X : np.ndarray
            
        Returns:
        -------
        np.ndarray
            An array of shape (n_samples, 2) containing the lower and upper bounds of the prediction intervals.
        """
    
        # Predict on the test set
        y_pred_test = self.model.predict(X)

        # Compute prediction intervals
        lower_bounds = y_pred_test - self.q
        upper_bounds = y_pred_test + self.q

        return np.column_stack((lower_bounds, upper_bounds))


if __name__ == "__main__":
    X = np.random.randn(100, 10)
    y = np.random.randn(100)
    model = RandomForestRegressor()
    conformal = SplitConformal(model)
    conformal.fit(X, y)
    print(conformal.predict(X))
