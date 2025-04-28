import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import KFold
from tqdm import tqdm


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
        predictions[np.arange(len(X)), all_models[j].predict(X).astype(int)] += 1
        # Normalize predictions by dividing each row by its sum
        # This converts counts to probabilities
        row_sums = np.sum(predictions, axis=1, keepdims=True)
        normalized_predictions = predictions / row_sums
        # Apply threshold using self.gamma
        # If probability is greater than gamma, include in prediction set (1), otherwise exclude (0)
        prediction_sets = np.zeros_like(normalized_predictions)
        prediction_sets[normalized_predictions > gamma] = 1
        
        return prediction_sets 


def APS_calibration_oob(X,y,bootstrap_indices, all_models, n_classes, classes_per_bootstrap, alpha):
    """
    Args: 
        X: features of the calibration set
        y: labels of the calibration set
        bootstrap_indices: indices of the bootstrap samples
        bootstrap_models: dictionary of bootstrap models
        alpha: significance level
    """
    all_predictions = []
    print('number of classes', n_classes)

    for i,model in tqdm(enumerate(all_models)):
        predictions = np.full((len(X), n_classes),np.nan)
        bootstrap_preds = model.predict_proba(X[bootstrap_indices[i]])
        for j,idx in enumerate(bootstrap_indices[i]):
            predictions[idx,classes_per_bootstrap[i]] = bootstrap_preds[j]
        #predictions[np.arange(bootstrap_indices[i]),classes_per_bootstrap[i]] = model.predict_proba(X[bootstrap_indices[i]])
        all_predictions.append(predictions)
    
    stacked_predictions = np.dstack(all_predictions)
    avg_predictions = np.nanmean(stacked_predictions, axis=2)
    # Check if there are any NaN values in the predictions
    
    
    avg_predictions = softmax(avg_predictions)
    
    avg_predictions_sorted = np.sort(avg_predictions, axis=1)[:, ::-1]

    # Get the indices of the sorted predictions
    sorted_indices = np.argsort(-avg_predictions, axis=1)
    
    correct_indices = np.where(sorted_indices == y[:, np.newaxis])
    
    cum_prob = np.cumsum(avg_predictions_sorted, axis=1)
    cum_prob_till_correct = cum_prob[correct_indices]
    print(f'correct_indices: {correct_indices[1]}')
    top_k = np.quantile(correct_indices[1], 1 - alpha) # ith position in the sorted predictions
    gamma = np.quantile(cum_prob_till_correct, 1 - alpha)
    return gamma, top_k.astype(int)


def get_stacked_predictions(X, bootstrap_models, n_classes, classes_per_bootstrap):
    """
    Args: 
        X: features of the calibration set
        y: labels of the calibration set
        bootstrap_models: dictionary of bootstrap models
        alpha: significance level
    """
    all_predictions = []
    for i,model in tqdm(enumerate(bootstrap_models)):
        predictions = np.full((len(X), n_classes),np.nan)
        predictions[:,classes_per_bootstrap[i]] = model.predict_proba(X)
        all_predictions.append(predictions)
    stacked_predictions = np.dstack(all_predictions)

    return stacked_predictions


def APS_calibration(X, y, bootstrap_models, n_classes, classes_per_bootstrap, alpha):
    """
    Args: 
        X: features of the calibration set
        y: labels of the calibration set
        bootstrap_models: dictionary of bootstrap models
        alpha: significance level
    """

    stacked_predictions = get_stacked_predictions(X, bootstrap_models, n_classes, classes_per_bootstrap)

    avg_predictions = np.mean(stacked_predictions, axis=2)
    avg_predictions = softmax(avg_predictions)
    


    avg_predictions_sorted = np.sort(avg_predictions, axis=1)[:, ::-1]
    
    # Get the indices of the sorted predictions
    sorted_indices = np.argsort(-avg_predictions, axis=1)

    # correct_indices 
    correct_indices = np.where(sorted_indices == y[:, np.newaxis])


    cum_prob = np.cumsum(avg_predictions_sorted, axis=1)
    cum_prob_till_correct = cum_prob[correct_indices]
    gamma = np.quantile(cum_prob_till_correct, 1 - alpha)
    top_k = np.quantile(correct_indices[1], 1 - alpha)
    print(f'gamma: {gamma}, top_k: {top_k}')
    
    return gamma, top_k.astype(int)


def APS_calibration_mv(X, y, bootstrap_models, n_classes, classes_per_bootstrap, alpha, quantile=True):


    m = len(bootstrap_models)
    n = len(X)
    assert len(y) == n

    stacked_predictions = get_stacked_predictions(X, bootstrap_models, n_classes, classes_per_bootstrap)
    assert stacked_predictions.shape == (n, n_classes, m)
    
    if quantile:
        stacked_predictions = np.sort(stacked_predictions, axis=2)[:, :, :]

    pred_data = {
        'sorted_indices': [],
        'cum_prob': [],
    }
    cum_prob_till_corrects = []

    
    for i in range(m):
        predictions = stacked_predictions[:, :, i]
        predictions_sorted = np.sort(predictions, axis=1)[:, ::-1]
        assert predictions_sorted.shape == (n, n_classes)

        # Get the indices of the sorted predictions
        sorted_indices = np.argsort(-predictions, axis=1)
        assert sorted_indices.shape == (n, n_classes)
        pred_data['sorted_indices'].append(sorted_indices)

        # correct_indices
        correct_indices = np.where(sorted_indices == y[:, np.newaxis])

        cum_prob = np.cumsum(predictions_sorted, axis=1)
        pred_data['cum_prob'].append(cum_prob)
        cum_prob_till_correct = cum_prob[correct_indices]
        cum_prob_till_corrects.append(cum_prob_till_correct)
        # gamma = np.quantile(cum_prob_till_correct, 1 - alpha)
        # top_k = np.quantile(correct_indices[1], 1 - alpha)
        # gammas.append(gamma)
        # top_ks.append(top_k.astype(int))
    
    # optimization loop over the target APS level, to achieve error alpha when taking
    # majority vote
    coverages = []
    gammass = []
    # start level at 1 - alpha/2, where MV is guaranteed to achieve error <= alpha
    level = 1 - alpha/2
    # we will slowly decrease the level until empirical error is close to alpha
    for i in range(40):
        gammas = [np.quantile(cum_prob_till_corrects[i], level) for i in range(m)]
        prediction_sets = predict_APS_calibration_mv(X, bootstrap_models, gammas, n_classes, classes_per_bootstrap,
                                                     top_ks=None, quantile=True, pred_data=pred_data)
        coverage = np.mean(prediction_sets[np.arange(n), y])
        print(f'level: {level}, coverage: {coverage}')
        if coverage < (1 - alpha):
            level = level + alpha/20
            break
        else:
            coverages.append(coverage)
            gammass.append(gammas)

            level = level - alpha/20
    
    print(f'chosen level: {level}, coverage: {coverages[-1]}')
    return gammass[-1], None # top_ks[best_idx]

    
def predict_APS_calibration(X, bootstrap_models, gamma, n_classes, classes_per_bootstrap, top_k):
    """
    Args: 
        X: features of the calibration set
        y: labels of the calibration set
        bootstrap_models: dictionary of bootstrap models
        alpha: significance level
    """

    stacked_predictions = get_stacked_predictions(X, bootstrap_models, n_classes, classes_per_bootstrap)

    avg_predictions = np.nanmean(stacked_predictions, axis=2)
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
        count = 0
        cum_prob_threshold_row = cum_prob_threshold[i]
        sorted_indices_row = sorted_indices[i]
        for j in range(n_classes):
            if cum_prob_threshold_row[j]:
                prediction_sets[i, sorted_indices_row[j]] = 1
                count += 1
            # if count >= top_k:
            #     break
    return prediction_sets


def predict_APS_calibration_mv(X, bootstrap_models, gammas, n_classes, classes_per_bootstrap,
                               top_ks, quantile=True, pred_data=None):

    if pred_data is None:
        stacked_predictions = get_stacked_predictions(X, bootstrap_models, n_classes, classes_per_bootstrap)

        if quantile:
            stacked_predictions = np.sort(stacked_predictions, axis=2)[:, :, :]
    
    prediction_setss = []
    m = len(bootstrap_models)
    n = len(X)

    for i in range(m):
        if pred_data is None:
            predictions = stacked_predictions[:, :, i]
            avg_predictions_sorted = np.sort(predictions, axis=1)[:, ::-1]
            sorted_indices = np.argsort(-predictions, axis=1) # descending order of predicted probabilities 
            
            # Get cumulative probabilities
            cum_prob = np.cumsum(avg_predictions_sorted, axis=1)

        else:
            sorted_indices = pred_data['sorted_indices'][i]
            cum_prob = pred_data['cum_prob'][i]
        cum_prob_threshold = np.where(cum_prob < gammas[i], True, False)
        # Create prediction sets
        prediction_sets = np.zeros((n, n_classes))
        for i in range(len(X)):
            count = 0
            cum_prob_threshold_row = cum_prob_threshold[i]
            sorted_indices_row = sorted_indices[i]
            for j in range(n_classes):
                if cum_prob_threshold_row[j]:
                    prediction_sets[i, sorted_indices_row[j]] = 1
                    count += 1
        prediction_setss.append(prediction_sets)
    
    # take majority vote
    prediction_sets = (np.dstack(prediction_setss).mean(axis=2) >= 0.5).astype(int)
    return prediction_sets



def temperature_scaling(logits, y_true, cv=5):
    pass






# def predict_APS_calibration(X, bootstrap_models, gamma, n_classes, classes_per_bootstrap, top_k):
#     """
#     Args: 
#         X: features of the calibration set
#         y: labels of the calibration set
#         bootstrap_models: dictionary of bootstrap models
#         alpha: significance level
#     """
#     all_predictions = []
#     for i,model in tqdm(enumerate(bootstrap_models)):
#         predictions = np.full((len(X), n_classes),np.nan)
#         predictions[:,classes_per_bootstrap[i]] = model.predict_proba(X)
#         all_predictions.append(predictions)
    
#     all_predictions = np.dstack(all_predictions)
#     avg_predictions = np.mean(all_predictions, axis=2)
#     avg_predictions = softmax(avg_predictions)
    
#     # Sort predictions, get class indices in descending order of predicted probabilities
#     avg_predictions_sorted = np.sort(avg_predictions, axis=1)[:, ::-1]
#     sorted_indices = np.argsort(-avg_predictions, axis=1) # descending order of predicted probabilities 
    
#     # Get cumulative probabilities
#     cum_prob = np.cumsum(avg_predictions_sorted, axis=1)
#     cum_prob_threshold = np.where(cum_prob < gamma, True, False)
    
#     # Create prediction sets
#     prediction_sets = np.zeros_like(avg_predictions)
#     for i in range(len(X)):
#         count = 0
#         cum_prob_threshold_row = cum_prob_threshold[i]
#         sorted_indices_row = sorted_indices[i]
#         for j in range(n_classes):
#             if cum_prob_threshold_row[j]:
#                 prediction_sets[i, sorted_indices_row[j]] = 1
#                 count += 1
#             if count == top_k - 1:
#                 break
#     return prediction_sets
