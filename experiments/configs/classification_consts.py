from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier

DATASETS = [
    "data_chess",
    "data_cover_type",
    "data_dionis",
    "data_helena",
    "data_isolet",
    "data_walking"
]

MODELS = {
    "LogisticRegression": LogisticRegressionCV(cv = 3, n_jobs = -1),
    "RandomForest": RandomForestClassifier(min_samples_leaf = 2, max_features = "sqrt", n_estimators = 100, random_state = 42, n_jobs = -1),
    "ExtraTrees": ExtraTreesClassifier(min_samples_leaf = 2, max_features = "sqrt", n_estimators = 100, random_state = 42, n_jobs = -1),
    "AdaBoost": AdaBoostClassifier(random_state = 42),
    "XGBoost": XGBClassifier(random_state = 42, n_jobs = -1),
    "MLP": MLPClassifier(random_state = 42, hidden_layer_sizes = (64,))}

TEST_MODELS = {"XGBoost": XGBClassifier(random_state = 42)}

VALID_UQ_METHODS = [
    'split_conformal_raps',
    'split_conformal_lac',
    'split_conformal_naive',
    'split_conformal_aps',
    'pcs_uq',
    'pcs_oob'
]

VALID_ESTIMATORS = [
    'XGBoost',
    'RandomForest',
    'ExtraTrees',
    'AdaBoost',
    'LogisticRegression',
    'MLP'
]

SINGLE_CONFORMAL_METHODS = [
    'split_conformal_raps',
    'split_conformal_lac',
    'split_conformal_naive',
    'split_conformal_aps',
]
