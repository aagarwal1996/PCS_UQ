import numpy as np


def get_coverage(y_true, y_pred):
    return np.mean(y_true >= y_pred[:, 0] and y_true <= y_pred[:, 1])

def get_mean_width(y_true, y_pred):
    return np.mean(y_pred[:, 1] - y_pred[:, 0])

