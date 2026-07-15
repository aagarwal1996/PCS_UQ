from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

DATASETS = [
    "data_chess",
    "data_cover_type",
    "data_dionis",
    "data_isolet",
    "data_language",
    "data_yeast",
]

MODELS = {
    "LogisticRegression": LogisticRegression(),
    "RandomForest": RandomForestClassifier(
        random_state=42, min_samples_leaf=5, n_jobs=-1
    ),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "MLP": MLPClassifier(random_state=42, hidden_layer_sizes=(64,)),
    "XGBoost": XGBClassifier(random_state=42, n_jobs=-1),
}

VALID_UQ_METHODS = [
    "split_conformal_raps",
    "split_conformal_aps",
    "split_conformal_topk",
    "majority_vote",
    "pcs_oob",
]

VALID_ESTIMATORS = [
    "RandomForest",
    "ExtraTrees",
    "AdaBoost",
    "LogisticRegression",
    "HistGradientBoosting",
    "MLP",
    "XGBoost",
]

SINGLE_CONFORMAL_METHODS = [
    "split_conformal_raps",
    "split_conformal_aps",
    "split_conformal_topk",
]
