from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor

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
    "Lasso": LassoCV(max_iter = 1000, cv = 3),
    "ElasticNet": ElasticNetCV(max_iter = 1000, cv = 3),
    "RandomForest": RandomForestRegressor(min_samples_leaf = 5, max_features = 0.33, n_estimators = 100, random_state = 42),
    "ExtraTrees": ExtraTreesRegressor(min_samples_leaf = 5, max_features = 0.33, n_estimators = 100, random_state = 42),
    "AdaBoost": AdaBoostRegressor(random_state = 42),
    "XGBoost": XGBRegressor(random_state = 42),
    "MLP": MLPRegressor(max_iter = 5000, random_state = 42),
}

TEST_MODELS = {"XGBoost": XGBRegressor(random_state = 42)}

VALID_UQ_METHODS = [
    'split_conformal',
    'studentized_conformal', 
    'majority_vote',
    'LocalConformalRegressor',
    'pcs_uq',
    'pcs_oob'
]

VALID_ESTIMATORS = [
    'XGBoost',
    'RandomForest',
    'ExtraTrees',
    'AdaBoost',
    'OLS',
    'Ridge',
    'Lasso',
    'ElasticNet',
    'MLP'
]

SINGLE_CONFORMAL_METHODS = [
    'split_conformal',
    'studentized_conformal', 
    'LocalConformalRegressor',
]
