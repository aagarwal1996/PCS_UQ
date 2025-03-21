from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier

DATASETS = [
    "data_chess",
    "data_cover_type",
    "data_dionis",
    "data_isolet",
    "data_language",
    "data_yeast"
]

MODELS = {
    "LogisticRegression": LogisticRegression(),
    "RandomForest": RandomForestClassifier(random_state = 42, min_samples_leaf = 5, n_jobs = 64),
    "ExtraTrees": ExtraTreesClassifier(random_state = 42, min_samples_leaf = 5, n_jobs = 64),
    "AdaBoost": AdaBoostClassifier(random_state = 42),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state = 42, max_depth = 4),
    "MLP": MLPClassifier(random_state = 42, hidden_layer_sizes = (64,))
}

VALID_UQ_METHODS = [
    'split_conformal_raps',
    'split_conformal_aps',
    'split_conformal_topk',
    'majority_vote',
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
