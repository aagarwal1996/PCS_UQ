import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor

from src.conformal_methods.regression.split_conformal import SplitConformal

def get_coverage(y_true, y_pred):
    coverage_indicator = (y_true >= y_pred[:, 0]) & (y_true <= y_pred[:, 1]).astype(int)
    return np.mean(coverage_indicator)

def get_mean_width(y_true, y_pred, return_scaled=False):
    mean_width = np.mean(np.abs(y_pred[:, 1] - y_pred[:, 0]))
    if return_scaled:
        scaled_mean_width = mean_width / np.abs(y_true.max() - y_true.min())
        return scaled_mean_width
    return mean_width

def get_median_width(y_true, y_pred, return_scaled=False):
    median_width = np.median(np.abs(y_pred[:, 1] - y_pred[:, 0]))
    if return_scaled:
        scaled_median_width = median_width / np.abs(y_true.max() - y_true.min())
        return scaled_median_width
    return median_width

def get_all_metrics(y_true, y_pred):
    return {
        'coverage': get_coverage(y_true, y_pred),
        'mean_width': get_mean_width(y_true, y_pred, return_scaled=False),
        'median_width': get_median_width(y_true, y_pred, return_scaled=False),
        'mean_width_scaled': get_mean_width(y_true, y_pred, return_scaled=True),
        'median_width_scaled': get_median_width(y_true, y_pred, return_scaled=True),
    }

if __name__ == "__main__":
    X, y = make_regression(n_samples=100, n_features=10, noise=10)
    conformal = SplitConformal(RandomForestRegressor())
    conformal.fit(X, y)
    y_pred = conformal.predict(X)
    print(get_all_metrics(y, y_pred))