import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import log_loss


def temperature_scaling(y_pred, y_true, temp_grid=None, t_min=0.1, t_max=5.0, n_temps=50):
    """
    Perform temperature scaling on the predicted probabilities using grid search.
    
    Temperature scaling is a simple post-processing calibration method that
    divides the logits by a temperature parameter before applying softmax.
    
    Parameters:
    -----------
    y_pred : numpy.ndarray
        Predicted probabilities from the model, shape (n_samples, n_classes)
    y_true : numpy.ndarray
        True labels (one-hot encoded or class indices), shape (n_samples, n_classes) or (n_samples,)
    temp_grid : list or numpy.ndarray, optional
        Grid of temperature values to search over. If None, a grid will be created using t_min, t_max, and n_temps.
    t_min : float, optional
        Minimum temperature value for the grid. Default is 0.1.
    t_max : float, optional
        Maximum temperature value for the grid. Default is 5.0.
    n_temps : int, optional
        Number of temperature values in the grid. Default is 50.
    
    Returns:
    --------
    tuple
        (calibrated_probs, optimal_temperature)
        - calibrated_probs: Calibrated probabilities after temperature scaling
        - optimal_temperature: The optimal temperature value found
    """
    # Convert y_true to one-hot encoding if it's not already
    if len(y_true.shape) == 1:
        n_classes = y_pred.shape[1]
        y_true_one_hot = np.zeros((len(y_true), n_classes))
        y_true_one_hot[np.arange(len(y_true)), y_true.astype(int)] = 1
        y_true = y_true_one_hot
    
    # Set default temperature grid if not provided
    if temp_grid is None:
        temp_grid = np.linspace(t_min, t_max, n_temps)
    
    # Convert probabilities to logits
    # Add small epsilon to avoid log(0)
    eps = 1e-12
    logits = np.log(y_pred + eps) - np.log(1 - y_pred + eps)
    
    # Initialize variables to track best temperature
    best_loss = float('inf')
    optimal_temperature = 1.0
    
    # Grid search for the best temperature
    for temperature in temp_grid:
        # Apply temperature scaling
        scaled_logits = logits / temperature
        # Convert back to probabilities using softmax
        scaled_probs = softmax(scaled_logits)
        # Calculate loss
        loss = log_loss(y_true, scaled_probs)
        
        # Update best temperature if this one is better
        if loss < best_loss:
            best_loss = loss
            optimal_temperature = temperature
    
    # Apply the optimal temperature to get calibrated probabilities
    calibrated_logits = logits / optimal_temperature
    calibrated_probs = softmax(calibrated_logits)
    
    return calibrated_probs, optimal_temperature


def softmax(logits):
    """
    Apply softmax function to convert logits to probabilities.
    
    Parameters:
    -----------
    logits : numpy.ndarray
        Input logits, shape (n_samples, n_classes)
    
    Returns:
    --------
    numpy.ndarray
        Probabilities after softmax, shape (n_samples, n_classes)
    """
    # Subtract max for numerical stability
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def model_prop_calibration(X_calib, y_calib, bootstrap_models, alpha=0.1):
    """
    Args: 
        X_calib: features of the calibration set
        y_calib: labels of the calibration set
        bootstrap_models: dictionary of bootstrap models
        alpha: significance level
    """
    model_list = [bootstrap_models[model] for model in bootstrap_models]
    all_models = []
    for model in model_list:
        for j in range(len(model)):
            all_models.append(model[j])
    # Get all predictions from all models
    all_predictions = []
    for model in all_models:
        all_predictions.append(model.predict(X_calib))
    
    # Get fraction of correct predictions for each sample
    scale_proportions = np.zeros(len(y_calib))
    for i in range(len(y_calib)):
        for j in range(len(all_predictions)):
            if y_calib[i] == all_predictions[j][i]:
                scale_proportions[i] += 1
    scale_proportions = scale_proportions / len(all_predictions)
    q = np.quantile(scale_proportions, alpha)
    return q
