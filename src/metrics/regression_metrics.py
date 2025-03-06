import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor

from src.conformal_methods.regression.split_conformal import SplitConformal

def get_coverage(y_true, y_pred):
    coverage_indicator = (y_true >= y_pred[:, 0]) & (y_true <= y_pred[:, 1]).astype(int)
    return np.mean(coverage_indicator)

def get_mean_width(y_pred):
    return np.mean(np.abs(y_pred[:, 1] - y_pred[:, 0]))

def get_all_metrics(y_true, y_pred):
    return {
        'coverage': get_coverage(y_true, y_pred),
        'mean_width': get_mean_width(y_pred),
    }

if __name__ == "__main__":
    X, y = make_regression(n_samples=100, n_features=10, noise=10)
    conformal = SplitConformal(RandomForestRegressor())
    conformal.fit(X, y)
    y_pred = conformal.predict(X)
    print(get_all_metrics(y, y_pred))