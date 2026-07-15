import pandas as pd
import pickle
import numpy as np

# PCS imports
from src.PCS.regression.pcs_uq import PCS_UQ
from src.PCS.regression.pcs_oob import PCS_OOB

# Conformal prediction imports
from src.conformal_methods.regression.split_conformal import SplitConformal
from src.conformal_methods.regression.studentized_conformal import StudentizedConformal
from src.conformal_methods.regression.local_conformal import LocalConformalRegressor
from src.conformal_methods.regression.majority_vote import MajorityVote
from src.conformal_methods.regression.jackknife_bootstrap import JackknifeBootstrap
from src.conformal_methods.regression.ensemble_conformal import (
    PredCheckedEnsembleSplitConformal,
    PredCheckedEnsembleStudentizedConformal,
    PredCheckedEnsembleJackknifeBootstrap,
)

from experiments.configs.regression_consts import (
    MODELS,
    DATASETS,
    VALID_UQ_METHODS,
    VALID_ESTIMATORS,
    SINGLE_CONFORMAL_METHODS,
    SINGLE_MODEL_METHODS,
    TEST_MODELS,
)


def get_conformal_methods(conformal_type, model_name="XGBoost", seed=0):
    # regular methods
    if conformal_type == "split_conformal":
        return SplitConformal(
            model=MODELS[model_name], seed=seed
        ), f"split_conformal_{model_name}"
    elif conformal_type == "studentized_conformal":
        return StudentizedConformal(
            mean_model=MODELS[model_name], sd_model=MODELS[model_name], seed=seed
        ), f"studentized_conformal_{model_name}"
    elif conformal_type == "LocalConformalRegressor":
        return LocalConformalRegressor(
            model=MODELS[model_name], seed=seed
        ), f"local_conformal_{model_name}"
    elif conformal_type == "majority_vote":
        return MajorityVote(models=MODELS, seed=seed), f"majority_vote"
    elif conformal_type == "jackknife_bootstrap":
        return JackknifeBootstrap(
            num_bootstraps=1000, model=MODELS[model_name], seed=seed
        ), f"jackknife_bootstrap_{model_name}"
    # ensemble algos
    elif conformal_type == "split_conformal_ensemble":
        return PredCheckedEnsembleSplitConformal(
            model=MODELS, seed=seed
        ), f"split_conformal_ensemble"
    elif conformal_type == "studentized_conformal_ensemble":
        return PredCheckedEnsembleStudentizedConformal(
            mean_model=MODELS, sd_model=MODELS, seed=seed, val_set="proper"
        ), f"studentized_conformal_ensemble"
    elif conformal_type == "jackknife_bootstrap_ensemble":
        return PredCheckedEnsembleJackknifeBootstrap(
            num_bootstraps=1000, model=MODELS, seed=seed, val_set="proper"
        ), f"jackknife_bootstrap_ensemble"
    # alt split
    elif conformal_type == "split_conformal_alt":
        return SplitConformal(
            model=MODELS[model_name], seed=seed, val_size=0.25
        ), f"split_conformal_alt_{model_name}"
    elif conformal_type == "studentized_conformal_alt":
        return StudentizedConformal(
            mean_model=MODELS[model_name],
            sd_model=MODELS[model_name],
            seed=seed,
            val_size=0.25,
        ), f"studentized_conformal_alt_{model_name}"
    elif conformal_type == "majority_vote_alt":
        return MajorityVote(
            models=MODELS, seed=seed, val_size=0.25
        ), f"majority_vote_alt"
    # ensemble + alt split
    elif conformal_type == "split_conformal_ensemble_alt":
        return PredCheckedEnsembleSplitConformal(
            model=MODELS, seed=seed, val_size=0.25
        ), f"split_conformal_ensemble_alt"
    elif conformal_type == "studentized_conformal_ensemble_alt":
        return PredCheckedEnsembleStudentizedConformal(
            mean_model=MODELS, sd_model=MODELS, seed=seed, val_size=0.25
        ), f"studentized_conformal_ensemble_alt"
    else:
        raise ValueError(f"Invalid conformal method: {conformal_type}")


def get_pcs_methods(pcs_type, seed=0, model_names=None):
    if pcs_type == "pcs_uq":
        return PCS_UQ(
            models=MODELS,
            num_bootstraps=1000,
            alpha=0.1,
            top_k=1,
            load_models=False,
            seed=seed,
        )
    elif pcs_type == "pcs_oob":
        return PCS_OOB(
            models=MODELS,
            num_bootstraps=1000,
            alpha=0.1,
            top_k=1,
            load_models=False,
            seed=seed,
        )
    elif pcs_type == "pcs_oob_downsample":
        return PCS_OOB(
            models=MODELS,
            num_bootstraps=1000,
            alpha=0.1,
            top_k=1,
            load_models=False,
            seed=seed,
            disturb_method="downsample",
            downsample_ratio=0.5,
        )
    elif pcs_type == "pcs_uq_alt":
        return PCS_UQ(
            models=MODELS,
            num_bootstraps=1000,
            alpha=0.1,
            top_k=1,
            load_models=False,
            seed=seed,
            val_size=0.5,
        )
    elif pcs_type == "pcs_oob_fixed_method":
        if model_names is None:
            raise ValueError("model_name is required for pcs_oob_fixed_method")
        models = {model_name: MODELS[model_name] for model_name in model_names}
        return PCS_OOB(
            models=models,
            num_bootstraps=1000,
            alpha=0.1,
            top_k=len(model_names),
            load_models=False,
            seed=seed,
        )
    elif pcs_type == "pcs_oob_downsample_fixed_method":
        if model_names is None:
            raise ValueError(
                "model_name is required for pcs_oob_downsample_fixed_method"
            )
        models = {model_name: MODELS[model_name] for model_name in model_names}
        return PCS_OOB(
            models=models,
            num_bootstraps=1000,
            alpha=0.1,
            top_k=len(model_names),
            load_models=False,
            seed=seed,
            disturb_method="downsample",
            downsample_ratio=0.5,
        )
    else:
        raise ValueError(f"Invalid PCS method: {pcs_type}")


def get_regression_datasets(dataset_name):
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Dataset '{dataset_name}' not found. Available datasets are: {DATASETS}"
        )

    X = pd.read_csv(f"experiments/data/{dataset_name}/X.csv")
    y = np.loadtxt(f"experiments/data/{dataset_name}/y.csv")
    with open(f"experiments/data/{dataset_name}/bin_df.pkl", "rb") as f:
        bin_df = pickle.load(f)
    importance = pd.read_csv(f"experiments/data/{dataset_name}/importances.csv")
    return X, y, bin_df, importance
