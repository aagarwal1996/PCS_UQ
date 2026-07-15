from sklearn.linear_model import LinearRegression, RidgeCV
from celer import LassoCV, ElasticNetCV
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor,
)
from xgboost import XGBRegressor
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
    "data_energy_efficiency",
    "data_kin8nm",
    "data_naval_propulsion",
    "data_diamond",
    "data_superconductor",
    "data_ca_housing",
    "data_elevator",
    "data_protein_structure",
    "data_debutanizer",
]

MODELS = {
    "OLS": LinearRegression(n_jobs=-1),
    "Ridge": RidgeCV(),
    "Lasso": LassoCV(cv=3, n_jobs=-1),
    "ElasticNet": ElasticNetCV(cv=3, n_jobs=-1),
    "RandomForest": RandomForestRegressor(
        min_samples_leaf=5,
        max_features=0.33,
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    ),
    "ExtraTrees": ExtraTreesRegressor(
        min_samples_leaf=5,
        max_features=0.33,
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    ),
    "AdaBoost": AdaBoostRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42, n_jobs=-1),
    "MLP": MLPRegressor(random_state=42, hidden_layer_sizes=(64,)),
}

TEST_MODELS = {"XGBoost": XGBRegressor(random_state=42)}

VALID_UQ_METHODS = [
    "split_conformal",
    "split_conformal_ensemble",
    "split_conformal_ensemble_alt",
    "split_conformal_alt",
    "studentized_conformal",
    "studentized_conformal_ensemble",
    "studentized_conformal_ensemble_alt",
    "studentized_conformal_alt",
    "majority_vote",
    "majority_vote_alt",
    "LocalConformalRegressor",
    "jackknife_bootstrap",
    "jackknife_bootstrap_ensemble",
    "pcs_uq",
    "pcs_uq_alt",
    "pcs_oob",
    "pcs_oob_downsample",
    "pcs_oob_fixed_method",
    "pcs_oob_downsample_fixed_method",
]

VALID_ESTIMATORS = [
    "XGBoost",
    "RandomForest",
    "ExtraTrees",
    "AdaBoost",
    "OLS",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "MLP",
]

SINGLE_CONFORMAL_METHODS = [
    "split_conformal",
    "split_conformal_alt",
    "studentized_conformal",
    "studentized_conformal_alt",
    "LocalConformalRegressor",
    "jackknife_bootstrap",
]

SINGLE_MODEL_METHODS = [
    "split_conformal",
    "split_conformal_alt",
    "studentized_conformal",
    "studentized_conformal_alt",
    "LocalConformalRegressor",
    "jackknife_bootstrap",
    "pcs_oob_fixed_method",
    "pcs_oob_downsample_fixed_method",
]
