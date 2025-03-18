import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import KFold

# def temperature_scaling_cv(y_pred, y_true, temp_grid=None, t_min=1e-4, t_max=5.0, n_temps=500):
#     """
#     Perform temperature scaling on the predicted probabilities using grid search.
    
#     Temperature scaling is a simple post-processing calibration method that
#     divides the logits by a temperature parameter before applying softmax.
    
#     Parameters:
#     -----------
#     y_pred : numpy.ndarray
#         Predicted probabilities from the model, shape (n_samples, n_classes)
#     y_true : numpy.ndarray
#         True labels (one-hot encoded or class indices), shape (n_samples, n_classes) or (n_samples,)
#     temp_grid : list or numpy.ndarray, optional
#         Grid of temperature values to search over. If None, a grid will be created using t_min, t_max, and n_temps.
#     t_min : float, optional
#         Minimum temperature value for the grid. Default is 0.1.
#     t_max : float, optional
#         Maximum temperature value for the grid. Default is 5.0.
#     n_temps : int, optional
#         Number of temperature values in the grid. Default is 50.
    
#     Returns:
#     --------
#     tuple
#         (calibrated_probs, optimal_temperature)
#         - calibrated_probs: Calibrated probabilities after temperature scaling
#         - optimal_temperature: The optimal temperature value found
#     """
#     # Convert y_true to one-hot encoding if it's not already
#     if len(y_true.shape) == 1:
#         n_classes = y_pred.shape[1]
#         y_true_one_hot = np.zeros((len(y_true), n_classes))
#         y_true_one_hot[np.arange(len(y_true)), y_true.astype(int)] = 1
#         y_true = y_true_one_hot
    
#     # Set default temperature grid if not provided
#     if temp_grid is None:
#         temp_grid = np.linspace(t_min, t_max, n_temps)
    
#     # Convert probabilities to logits
#     # Add small epsilon to avoid log(0)
#     eps = 1e-12
#     logits = np.log(y_pred + eps) - np.log(1 - y_pred + eps)
    
#     # Initialize variables to track best temperature
#     best_loss = float('inf')
#     optimal_temperature = 1.0
    
#     # Grid search for the best temperature
#     for temperature in temp_grid:
#         # Apply temperature scaling
#         scaled_logits = logits / temperature
#         # Convert back to probabilities using softmax
#         scaled_probs = softmax(scaled_logits)
#         # Calculate loss
#         loss = log_loss(y_true, scaled_probs)
        
#         # Update best temperature if this one is better
#         if loss < best_loss:
#             best_loss = loss
#             optimal_temperature = temperature
    
#     # Apply the optimal temperature to get calibrated probabilities
#     calibrated_logits = logits / optimal_temperature
#     calibrated_probs = softmax(calibrated_logits)
    
#     return calibrated_probs, optimal_temperature


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
    #scale_proportions, temperature = temperature_scaling(scale_proportions, y_calib)
    temperature = 1.0
    q = np.quantile(scale_proportions, alpha)
    return q, temperature


def predict_model_prop_calibration(X, bootstrap_models, gamma, n_classes, temperature = 1.0):
    model_list = [bootstrap_models[model] for model in bootstrap_models]
    all_models = []
    for model in model_list:
            for j in range(len(model)): 
                all_models.append(model[j])
        # Get all predictions from all models
        
    predictions = np.zeros((len(X), n_classes))
    for j in range(len(all_models)):
        print(all_models[j].predict(X))
        predictions[np.arange(len(X)), all_models[j].predict(X)] += 1
        # Normalize predictions by dividing each row by its sum
        # This converts counts to probabilities
        row_sums = np.sum(predictions, axis=1, keepdims=True)
        normalized_predictions = predictions / row_sums
        # Apply threshold using self.gamma
        # If probability is greater than gamma, include in prediction set (1), otherwise exclude (0)
        prediction_sets = np.zeros_like(normalized_predictions)
        prediction_sets[normalized_predictions > gamma] = 1
        
        return prediction_sets 


def APS_calibration(X, y, bootstrap_models, alpha):
    """
    Args: 
        X: features of the calibration set
        y: labels of the calibration set
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
        all_predictions.append(model.predict_proba(X))
    
    all_predictions = np.dstack(all_predictions)
    avg_predictions = np.mean(all_predictions, axis=2)
    avg_predictions = softmax(avg_predictions)
    
    

    #avg_predictions, temperature = temperature_scaling(logits, y)
    temperature = 1.0

    avg_predictions_sorted = np.sort(avg_predictions, axis=1)[:, ::-1]
    
    # Get the indices of the sorted predictions
    sorted_indices = np.argsort(-avg_predictions, axis=1)

    # correct_indices 
    correct_indices = np.where(sorted_indices == y[:, np.newaxis])

    
    
    

    cum_prob = np.cumsum(avg_predictions_sorted, axis=1)
    cum_prob_till_correct = cum_prob[correct_indices]
    gamma = np.quantile(cum_prob_till_correct, 1 - alpha)
    print(gamma)
    
    return gamma, temperature
    


def predict_APS_calibration(X, bootstrap_models, gamma, n_classes, temperature):
    """
    Args: 
        X: features of the calibration set
        y: labels of the calibration set
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
        all_predictions.append(model.predict_proba(X))
    
    all_predictions = np.dstack(all_predictions)
    avg_predictions = np.mean(all_predictions, axis=2)
    avg_predictions = softmax(avg_predictions)
    
    # Sort predictions, get class indices in descending order of predicted probabilities
    avg_predictions_sorted = np.sort(avg_predictions, axis=1)[:, ::-1]
    sorted_indices = np.argsort(-avg_predictions, axis=1) # descending order of predicted probabilities 
    
    # Get cumulative probabilities
    cum_prob = np.cumsum(avg_predictions_sorted, axis=1)
    cum_prob_threshold = np.where(cum_prob < gamma, True, False)
    
    # Create prediction sets
    prediction_sets = np.zeros_like(avg_predictions)
    for i in range(len(X)):
        cum_prob_threshold_row = cum_prob_threshold[i]
        sorted_indices_row = sorted_indices[i]
        for j in range(n_classes):
            if cum_prob_threshold_row[j]:
                prediction_sets[i, sorted_indices_row[j]] = 1
    return prediction_sets


def multi_threshold_calibration(X, y, bootstrap_models, alpha):
    """
    Args: 
        X: features of the calibration set
        y: labels of the calibration set
        bootstrap_models: dictionary of bootstrap models
    """
    model_list = [bootstrap_models[model] for model in bootstrap_models]
    all_models = []
    for model in model_list:
        for j in range(len(model)):
            all_models.append(model[j])
    # Get all predictions from all models
    all_predictions = []
    for model in all_models:
        all_predictions.append(model.predict_proba(X))
    
    all_predictions = np.dstack(all_predictions) # shape (n_samples, n_classes, n_models)
    thresholds = np.quantile(all_predictions, 1 - alpha, axis=2)
    return thresholds, 1.0



def temperature_scaling(logits, y_true, cv=5):
    """
    Fit the temperature using cross-validation.
    
    Parameters:
    - logits: raw model outputs (pre-softmax) of shape (n_samples, n_classes)
    - y_true: true labels (one-hot encoded or class indices)
    - cv: number of cross-validation folds
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    best_temp = []

    def _loss(temp, logits, y_true):
        scaled_probs = softmax(logits / temp)
        loss = log_loss(y_true, scaled_probs)
        return loss

    for train_idx, val_idx in kf.split(logits):
        logits_train, logits_val = logits[train_idx], logits[val_idx]
        y_train, y_val = y_true[train_idx], y_true[val_idx]
        result = minimize(_loss, x0=[1.0], args=(logits_val, y_val))
        best_temp.append(result.x[0])

    temperature = np.mean(best_temp)  # Use average of CV temperatures 
    return softmax(logits / temperature), temperature