import numpy as np
import pandas as pd
import os
import pickle
import copy
from tqdm import tqdm
# Sklearn Imports
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier  
from sklearn.linear_model import RidgeClassifierCV, LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score, log_loss
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder


# PCS UQ Imports
from src.PCS.classification.calibration_utils import APS_calibration_oob, predict_APS_calibration
from src.metrics.classification_metrics import get_all_metrics
from src.PCS.classification.multi_class_pcs import MultiClassPCS


class MultiClassPCS_OOB(MultiClassPCS):
    def __init__(self, models, num_bootstraps=100, alpha=0.1, seed=42, top_k = 1, save_path = None, load_models = True, metric = log_loss, val_size = 0.25, calibration_method = 'model_prop'):
        """
        MultiClassPCS_OOB

        Args:
            models: dictionary of model names and models
            num_bootstraps: number of bootstraps
            alpha: significance level
            seed: random seed
            top_k: number of top models to use
            save_path: path to save the models
            load_models: whether to load the models from the save_path
            metric: metric to use for the prediction scores -- assume that higher is better
            calibration_method: calibration method to use
        """
        self.models = {model_name: copy.deepcopy(model) for model_name, model in models.items()}
        self.num_bootstraps = num_bootstraps
        self.alpha = alpha
        self.seed = seed
        self.top_k = top_k
        self.save_path = save_path
        self.load_models = load_models
        self.metric = metric
        self.val_size = val_size
        self.calibration_method = calibration_method
        self.n_classes = None
        self.pred_scores = {model: np.inf for model in self.models}
        
    def fit(self, X, y, alpha = 0.1):
        """
        Fit the models
        """
        self.n_classes = len(np.unique(y))
        le = LabelEncoder()
        y = le.fit_transform(y)
        self._label_encoder = le
        if alpha is None:
            alpha = self.alpha
        X_train, X_calib, y_train, y_calib = train_test_split(X, y, test_size=self.val_size, random_state=self.seed, stratify=y)
        self._train(X_train, y_train) # train the models such that they are ready for calibration, saved in self.models
        self._pred_check(X_calib, y_calib) # check the predictions of the models, saved in self.models
        self.top_k_models = self._get_top_k()
        self._train_top_k(X, y)
        self.gamma, self.temperature = self.calibrate(X, y)

    def _train_top_k(self, X, y):
        """
        Train the models and store out-of-bag indices
        """
        # Initialize dictionaries once outside the model loop
        
        self.bootstrap_models = {}
        self.oob_indices = {}
        self._flattened_bootstrap_models = []
        self._flattened_oob_indices = []
        self._classes_per_bootstrap = []
        
        for model_name, model in self.top_k_models.items():
            # Initialize lists for each model
            self.bootstrap_models[model_name] = []
            self.oob_indices[model_name] = []
            
            for i in tqdm(range(self.num_bootstraps), desc=f"Training {model_name} models"):
                bootstrap_seed = self.seed + i
                # Try to load existing bootstrap model and OOB indices if enabled
                model_path = f"{self.save_path}/pcs_oob/{model_name}_model_seed_{bootstrap_seed}.pkl" if self.save_path else None
                oob_path = f"{self.save_path}/pcs_oob/{model_name}_oob_seed_{bootstrap_seed}.pkl" if self.save_path else None
                bootstrap_model = None
                
                if self.load_models and model_path and os.path.exists(model_path) and os.path.exists(oob_path):
                    with open(model_path, "rb") as f:
                        bootstrap_model = pickle.load(f)
                    with open(oob_path, "rb") as f:
                        oob_indices = pickle.load(f)
                    self.oob_indices[model_name].append(oob_indices)
                    self._flattened_oob_indices.append(oob_indices)
                else:
                    # Bootstrap the data
                    n_samples = len(X)
                    class_counts = np.bincount(y.astype(int))
                    class_idx_to_freq  = {i: class_counts[i] / n_samples for i in range(len(class_counts))}
                    weights = np.ones(n_samples)
                    # for i in range(n_samples):
                    #     weights[i] = class_idx_to_freq[y[i]]
                    # weights = weights / weights.sum()
                    weights = weights / weights.sum()
                    bootstrap_indices = np.random.choice(range(n_samples), size=n_samples, replace=True, p=weights)
                    oob_indices = list(set(range(n_samples)) - set(bootstrap_indices))
                    
                    
                    X_boot = X[bootstrap_indices]
                    y_boot = y[bootstrap_indices]
                    self._classes_per_bootstrap.append(np.unique(y_boot))
                    
                    # Store OOB indices
                    self.oob_indices[model_name].append(oob_indices)
                    self._flattened_oob_indices.append(oob_indices)
                    # Create and fit bootstrap model
                    bootstrap_model = copy.deepcopy(model)
                    bootstrap_model.fit(X_boot, y_boot)
                    
                    # Save the bootstrap model and OOB indices if save path is provided
                    if model_path:
                        os.makedirs(os.path.dirname(model_path), exist_ok=True)
                        with open(model_path, "wb") as f:
                            pickle.dump(bootstrap_model, f)
                        with open(oob_path, "wb") as f:
                            pickle.dump(oob_indices, f)
                
                self.bootstrap_models[model_name].append(bootstrap_model)
                self._flattened_bootstrap_models.append(bootstrap_model)
            print(f"Finished training {model_name} models")

    def calibrate(self, X, y):
        if self.calibration_method == 'model_prop':
            return NotImplementedError("Model Propagation is not supported for OOB calibration")
        elif self.calibration_method == 'APS':
            return APS_calibration_oob(X, y, self._flattened_oob_indices, self._flattened_bootstrap_models, self.n_classes, self._classes_per_bootstrap, self.alpha)
        else:
            raise ValueError(f"Calibration method {self.calibration_method} not supported")
    
    def predict(self, X):
        if self.calibration_method == 'model_prop':
            return NotImplementedError("Model Propagation is not supported for OOB calibration")
        elif self.calibration_method == 'APS':
            return predict_APS_calibration(X = X, bootstrap_models = self._flattened_bootstrap_models, gamma = self.gamma, 
                                              n_classes = self.n_classes, classes_per_bootstrap = self._classes_per_bootstrap, top_k = self.temperature)
        else:
            raise ValueError(f"Calibration method {self.calibration_method} not supported")
    
    
        
if __name__ == "__main__":
    models = {
        'rf': RandomForestClassifier(n_estimators=5, random_state=42, min_samples_leaf=50),
        'lr': LogisticRegression(random_state=42)
    }
    X, y = make_classification(n_samples=250, n_features=10, n_classes=10, n_informative=5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    pcs_oob = MultiClassPCS_OOB(models, num_bootstraps=500, alpha=0.1, seed=42, top_k=1, save_path='./models', load_models=False, metric=log_loss, calibration_method='APS')
    pcs_oob.fit(X_train, y_train)
    intervals = pcs_oob.predict(X_test)
    print(get_all_metrics(y_test, intervals))