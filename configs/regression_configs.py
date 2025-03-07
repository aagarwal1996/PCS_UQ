import json
from pathlib import Path

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

MODELS = {
    "rf": RandomForestRegressor(),
    "lr": LinearRegression(),
    "ridge": RidgeCV(),
    "lasso": LassoCV(),
    "enet": ElasticNetCV(),
    "et": ExtraTreesRegressor(),
    "ada": AdaBoostRegressor(),
    "xgb": XGBRegressor(),
    "hgb": HistGradientBoostingRegressor(),
    "mlp": MLPRegressor(),
}

def get_conformal_methods(models):
    methods = {}
    for model_name, model in models.items():
        methods[f"split_conformal_{model_name}"] = SplitConformal(model=model)
        methods[f"studentized_conformal_{model_name}"] = StudentizedConformal(model=model)
        methods[f"local_conformal_{model_name}"] = LocalConformalRegressor(model=model)
    return methods

def get_pcs_methods(models):
    methods = {}
    pcs_uq = PCS_UQ(models=MODELS, num_bootstraps=1000, alpha=0.1, seed=42, top_k=1)
    pcs_oob = PCS_OOB(models=MODELS, num_bootstraps=1000, alpha=0.1, seed=42, top_k=1)
    return {
        "pcs_uq": pcs_uq,
        "pcs_oob": pcs_oob
    }
    
UQ_METHODS = {
    "pcs_uq": PCS_UQ,
    "pcs_oob": PCS_OOB,
    "split_conformal_ols": SplitConformal(model=LinearRegression()),
    "split_conformal_ridge": SplitConformal(model=RidgeCV()),
    "split_conformal_lasso": SplitConformal(model=LassoCV()),
    "split_conformal_enet": SplitConformal(model=ElasticNetCV()),
    "split_conformal_rf": SplitConformal(model=RandomForestRegressor()),
    "split_conformal_et": SplitConformal(model=ExtraTreesRegressor()),
    "split_conformal_ada": SplitConformal(model=AdaBoostRegressor()),
    "split_conformal_xgb": SplitConformal(model=XGBRegressor()),
    "split_conformal_hgb": SplitConformal(model=HistGradientBoostingRegressor()),
    
    "studentized_conformal": StudentizedConformal,
    "local_conformal": LocalConformalRegressor
}