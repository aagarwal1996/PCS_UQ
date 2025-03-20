import json
from pathlib import Path
import pandas as pd
import pickle
import numpy as np

# Model imports
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# PCS imports
from src.PCS.classification.multi_class_pcs import MultiClassPCS
from src.PCS.classification.multi_class_pcs_oob import MultiClassPCS_OOB

# Conformal prediction imports
from src.conformal_methods.classification.multi_class_conformal import MultiClassConformal


from experiments.configs.classification_consts import MODELS, DATASETS, VALID_UQ_METHODS, VALID_ESTIMATORS, SINGLE_CONFORMAL_METHODS

#MODELS = {"XGBoost": XGBRegressor(random_state = 42)}#, "RandomForest": RandomForestRegressor(min_samples_leaf = 5, max_features = 0.33, n_estimators = 100, random_state = 42)}
#MODELS = {"LightGBM": LGBMRegressor(random_state = 42)}


def get_conformal_methods(conformal_type, model_name= 'XGBoost', seed = 0):
    if conformal_type == "split_conformal_raps":
        return MultiClassConformal(model=MODELS[model_name], seed = seed, conformity_score = 'RAPS'), f"split_conformal_raps_{model_name}"
    elif conformal_type == "split_conformal_lac":
        return MultiClassConformal(model=MODELS[model_name], seed = seed, conformity_score = 'LAC'), f"split_conformal_lac_{model_name}"
    elif conformal_type == "split_conformal_naive":
        return MultiClassConformal(model=MODELS[model_name], seed = seed, conformity_score = 'Naive'), f"split_conformal_naive_{model_name}"
    elif conformal_type == "split_conformal_aps":
        return MultiClassConformal(model=MODELS[model_name], seed = seed, conformity_score = 'APS'), f"split_conformal_aps_{model_name}"
    elif conformal_type == "split_conformal_topk":
        return MultiClassConformal(model=MODELS[model_name], seed = seed, conformity_score = 'TopK'), f"split_conformal_topk_{model_name}"

def get_pcs_methods(pcs_type, seed = 0):
    if pcs_type == "pcs_uq":
        return MultiClassPCS(models=MODELS, num_bootstraps=500, alpha=0.1, top_k=1, load_models=False, seed = seed, calibration_method = 'APS')
    elif pcs_type == "pcs_oob":
        return MultiClassPCS_OOB(models=MODELS, num_bootstraps=500, alpha=0.1, top_k=1, load_models=False, seed = seed, calibration_method = 'APS')
    elif pcs_type == "pcs_uq_model_prop":
        return MultiClassPCS(models=MODELS, num_bootstraps=100, alpha=0.1, top_k=1, load_models=False, seed = seed, calibration_method = 'model_prop')
    elif pcs_type == "pcs_oob_model_prop":
        return MultiClassPCS_OOB(models=MODELS, num_bootstraps=1000, alpha=0.1, top_k=1, load_models=False, seed = seed, calibration_method = 'model_prop')
    else:
        raise ValueError(f"Invalid PCS method: {pcs_type}")



def get_classification_datasets(dataset_name):
    if dataset_name not in DATASETS:
        raise ValueError(f"Dataset '{dataset_name}' not found. Available datasets are: {DATASETS}")
    
    X = pd.read_csv(f"experiments/data/classification/{dataset_name}/X.csv")
    y = np.loadtxt(f"experiments/data/classification/{dataset_name}/y.csv")
    with open(f'experiments/data/classification/{dataset_name}/bin_df.pkl', 'rb') as f:
        bin_df = pickle.load(f)
    importance = pd.read_csv(f"experiments/data/classification/{dataset_name}/importances.csv")
    return X, y, bin_df, importance


