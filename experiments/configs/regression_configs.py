import json
from pathlib import Path
import pandas as pd
import pickle
# Model imports
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

# PCS imports
from src.PCS.regression.pcs_uq import PCS_UQ
from src.PCS.regression.pcs_oob import PCS_OOB

# Conformal prediction imports
from src.conformal_methods.regression.split_conformal import SplitConformal
from src.conformal_methods.regression.studentized_conformal import StudentizedConformal
from src.conformal_methods.regression.local_conformal import LocalConformalRegressor

DATASETS = [
    "data_parkinsons",
    "data_airfoil",
    "data_computer",
    "data_concrete",
    "data_powerplant",
    "data_miami_housing",
    "data_insurance",
    "data_qsar",
    "data_allstate",
    "data_mercedes",
    "data_energy_efficiency",
    "data_kin8nm",
    "data_naval_propulsion",
    "data_diamond",
    "data_superconductor",
    "data_ca_housing",
    "data_protein_structure",
]

MODELS = {
    "OLS": LinearRegression(),
    "Ridge": RidgeCV(),
    "Lasso": LassoCV(max_iter = 5000),
    "ElasticNet": ElasticNetCV(max_iter = 5000),
    "RandomForest": RandomForestRegressor(min_samples_leaf = 5, max_features = 0.33, n_estimators = 100, random_state = 42),
    "ExtraTrees": ExtraTreesRegressor(min_samples_leaf = 5, max_features = 0.33, n_estimators = 100, random_state = 42),
    "AdaBoost": AdaBoostRegressor(random_state = 42),
    "XGBoost": XGBRegressor(random_state = 42),
    "HistGradientBoosting": HistGradientBoostingRegressor(random_state = 42),
    "MLP": MLPRegressor(max_iter = 5000, random_state = 42),
}

def get_conformal_methods(models):
    methods = {}
    for model_name, model in models.items():
        methods[f"split_conformal_{model_name}"] = SplitConformal(model=model)
        methods[f"studentized_conformal_{model_name}"] = StudentizedConformal(mean_model=model, sd_model=model)
        methods[f"local_conformal_{model_name}"] = LocalConformalRegressor(model=model)
    return methods

def get_pcs_methods(models):
    methods = {}
    pcs_uq = PCS_UQ(models=MODELS, num_bootstraps=1000, alpha=0.1, top_k=1)
    pcs_oob = PCS_OOB(models=MODELS, num_bootstraps=1000, alpha=0.1, top_k=1)
    return {
        "pcs_uq": pcs_uq,
        "pcs_oob": pcs_oob
    }

def get_uq_methods(models):
    return get_conformal_methods(models) | get_pcs_methods(models)

def get_regression_datasets(dataset_name):
    if dataset_name not in DATASETS:
        raise ValueError(f"Dataset '{dataset_name}' not found. Available datasets are: {DATASETS}")
    
    X = pd.read_csv(f"../data/{dataset_name}/X.csv")
    y = pd.read_csv(f"../data/{dataset_name}/y.csv")
    with open(f'../data/{dataset_name}/bin_df.pk', 'rb') as f:
        bin_df = pickle.load(f)
    importance = pd.read_csv(f"../data/{dataset_name}/importance.csv")
    return X, y, bin_df

if __name__ == "__main__":
    print(get_uq_methods(MODELS))