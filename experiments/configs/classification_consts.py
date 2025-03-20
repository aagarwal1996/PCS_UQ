from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
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
    "RandomForest": RandomForestClassifier(n_jobs = -1, min_samples_leaf = 1),
    "ExtraTrees": ExtraTreesClassifier(n_jobs = -1, min_samples_leaf = 1),
    "AdaBoost": AdaBoostClassifier(random_state = 42),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state = 42),
    "MLP": MLPClassifier(random_state = 42, hidden_layer_sizes = (64,))
}

VALID_UQ_METHODS = [
    'split_conformal_raps',
    'split_conformal_aps',
    'split_conformal_topk',
    'pcs_uq',
    'pcs_oob',
]

VALID_ESTIMATORS = [
    'RandomForest',
    'ExtraTrees',
    'AdaBoost',
    'LogisticRegression',
    'HistGradientBoosting',
    'MLP'
]

SINGLE_CONFORMAL_METHODS = [
    'split_conformal_raps',
    'split_conformal_aps',
    'split_conformal_topk'
]
