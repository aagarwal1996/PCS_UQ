import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from copy import deepcopy


def get_coverage(y_true, y_pred, empty_set=None):
    if len(y_true) == 0:
        return np.nan
    y_true, y_pred = process_empty_set(y_true, y_pred, empty_set=empty_set)
    coverage_indicator = y_pred[np.arange(len(y_true)).astype(int), y_true.astype(int)]
    return np.mean(coverage_indicator)

def get_mean_width(y_true, y_pred, return_scaled=False, empty_set=None):
    if len(y_true) == 0:
        return np.nan
    y_true, y_pred = process_empty_set(y_true, y_pred, empty_set=empty_set)
    mean_width = np.mean(np.sum(y_pred, axis=1))
    if return_scaled:
        scaled_mean_width = mean_width / y_pred.shape[1]
        return scaled_mean_width
    return mean_width

def get_median_width(y_true, y_pred, return_scaled=False, empty_set=None):
    if len(y_true) == 0:
        return np.nan
    y_true, y_pred = process_empty_set(y_true, y_pred, empty_set=empty_set)
    median_width = np.median(np.sum(y_pred, axis=1))
    if return_scaled:
        scaled_median_width = median_width / y_pred.shape[1]
        return scaled_median_width
    return median_width

def get_class_mean_width(y_true, y_pred, return_scaled=False, empty_set=None):
    if len(y_true) == 0:
        return np.nan
    y_true, y_pred = process_empty_set(y_true, y_pred, empty_set=empty_set)
    widths = np.sum(y_pred, axis=1)
    class_mean_widths = []
    for cls in np.arange(y_pred.shape[1]).astype(int):
        class_mean_widths.append(np.mean(widths[y_true == cls]))
    class_mean_widths = np.array(class_mean_widths)
    if return_scaled:
        scaled_class_mean_widths = class_mean_widths / y_pred.shape[1]
        return scaled_class_mean_widths
    return class_mean_widths

def get_class_median_width(y_true, y_pred, return_scaled=False, empty_set=None):
    if len(y_true) == 0:
        return np.nan
    y_true, y_pred = process_empty_set(y_true, y_pred, empty_set=empty_set)
    widths = np.sum(y_pred, axis=1)
    class_median_widths = []
    for cls in np.arange(y_pred.shape[1]).astype(int):
        class_median_widths.append(np.median(widths[y_true == cls]))
    class_median_widths = np.array(class_median_widths)
    if return_scaled:
        scaled_class_median_widths = class_median_widths / y_pred.shape[1]
        return scaled_class_median_widths
    return class_median_widths

def get_all_metrics(y_true, y_pred, empty_set=None):
    return {
        'coverage': get_coverage(y_true, y_pred, empty_set=empty_set),
        'mean_width': get_mean_width(y_true, y_pred, return_scaled=False, empty_set=empty_set),
        'median_width': get_median_width(y_true, y_pred, return_scaled=False, empty_set=empty_set),
        'mean_width_scaled': get_mean_width(y_true, y_pred, return_scaled=True, empty_set=empty_set),
        'median_width_scaled': get_median_width(y_true, y_pred, return_scaled=True, empty_set=empty_set),
    }

def process_empty_set(y_true, y_pred, empty_set):
    if empty_set is None:
        return y_true, y_pred
    if empty_set == 'remove':
        non_empty_indicators = (y_pred.sum(axis=1) != 0)
        y_pred = y_pred[non_empty_indicators]
        y_true = y_true[non_empty_indicators]
    elif empty_set == 'to_full':
        y_pred = deepcopy(y_pred)
        y_pred[y_pred.sum(axis=1) == 0] = 1
    else:
        raise ValueError(f"empty_set must be either 'remove' or 'to_full', got {empty_set}")
    return y_true, y_pred

if __name__ == "__main__":
    # TODO: Add tests
    pass