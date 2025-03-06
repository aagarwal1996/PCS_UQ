
import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.base import clone
from src.conformal_methods.regression.split_conformal import SplitConformal
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
class Conformal_Majority_Vote:
    def __init__ (self, models, alpha = 0.1, seed = 42):
        """
        Initialize the Conformal Majority Vote model

        Args:
            models: dictionary of model names and models
            alpha: significance level
            seed: random seed
        """
        self.models = {model_name: clone(model) for model_name, model in models.items()}
        self.conformals = {model_name: None for model_name in models.keys()}
        self.K = len(models)
        self.alpha = alpha
        self.seed = seed
        self.trained = False


    def fit(self, X, y, alpha = None):
        """
        Fit the model on the training data.

        Args:
            X: training features
            y: training labels
        """
        if alpha is None:
            alpha = self.alpha
        self.conformals = {}
        X_train, X_calib, y_train, y_calib = train_test_split(X, y, test_size=0.5, random_state=self.seed)
        self._train(X_train, y_train)
        self._calibrate(X_calib, y_calib)
        return self
        
    def _train(self, X, y):
        for model_name, model in self.models.items():
            split_conformal_model = SplitConformal(model, self.alpha)
            split_conformal_model.fit(X, y)
            self.conformals[model_name] = split_conformal_model
        self.trained = True
    
    def _calibrate(self, X, y, alpha=0.1):
        if self.trained == False:
            raise ValueError("Model must be trained before calling calibrate.")
        qs = {}
        for model_name, conformal in self.conformals.items():
            q = conformal._calibrate(X, y, alpha/2)
            qs[model_name] = q

        self.calibrated = True
        self.qs = qs
        return qs
    
    def predict(self, X, tau = 0.5):
        """
        Predict the intervals for the test set.

        Args:
            X: test features
            tau: significance level
        """
        model_intervals = {}
        for model_name, conformal in self.conformals.items():
            model_intervals[model_name] = conformal.predict(X) # Shape: (n_samples, 2)
        lower_bounds = np.zeros((X.shape[0], self.K))
        upper_bounds = np.zeros((X.shape[0], self.K))
        all_bounds = np.zeros((X.shape[0], 2*self.K))
        for i, model_name in enumerate(self.conformals.keys()):
            lower_bounds[:, i] = model_intervals[model_name][:, 0]
            upper_bounds[:, i] = model_intervals[model_name][:, 1]
            all_bounds[:, 2*i] = lower_bounds[:, i]
            all_bounds[:, 2*i + 1] = upper_bounds[:, i]
        all_bounds = np.sort(all_bounds, axis=1)
        return all_bounds
        # self.tau = tau
    
    
if __name__ == "__main__":
    X,y = make_regression(n_samples=1000, n_features=10, noise=10)
    models = {
        'rf': RandomForestRegressor(),
        'lr': LinearRegression()
    }
    conformal_majority_vote = Conformal_Majority_Vote(models)
    conformal_majority_vote.fit(X, y)
    print(conformal_majority_vote.predict(X))



        # lb_cols = [f'{model_name}_lb' for model_name in self.conformals.keys()]
        # ub_cols = [f'{model_name}_ub' for model_name in self.conformals.keys()]
        # all_conformals = pd.DataFrame(np.zeros((x_test.shape[0], 2*self.K)), 
        #                               columns=lb_cols + ub_cols)
        # for model_name, conformal in self.conformals.items():
        #     single_conformal_pred = conformal.predict(x_test)
        #     all_conformals[f'{model_name}_lb'] = single_conformal_pred['lb']
        #     all_conformals[f'{model_name}_ub'] = single_conformal_pred['ub']
        # self.all_conformals = all_conformals
        # self.prediction_intervals = pd.DataFrame(all_conformals.apply(lambda row: self.get_majority_vote(row, self.K, self.tau),axis=1),
        #                                         columns=['intervals'])

        # self.predicted = True
        # return self.prediction_intervals

    # def get_majority_vote(self, row, K, tau):
    #     lower_bounds = row.iloc[:K].to_numpy()
    #     upper_bounds = row.iloc[K:].to_numpy()
    #     q = np.sort(row)
    #     i = 1
    #     lower = []
    #     upper = []
    #     while i < 2*K:
    #         cond_i = np.mean((lower_bounds <= (q[i-1] + q[i])/ 2) & (upper_bounds >= (q[i-1] + q[i])/ 2))
    #         if i == 10 and row.name == 1:
    #             print(lower_bounds, upper_bounds, q[i-1], q[i], (lower_bounds <= (q[i-1] + q[i])/ 2) & (upper_bounds >= (q[i-1] + q[i])/ 2))
    #         if cond_i > tau:
    #             lower.append(q[i-1])
    #             j = i
    #             cond_j = np.mean((lower_bounds <= (q[j-1] + q[j])/ 2) & (upper_bounds >= (q[j-1] + q[j])/ 2))
    #             while (cond_j > tau) and (j < 2*K):
    #                 j += 1
    #                 cond_j = np.mean((lower_bounds <= (q[j-1] + q[j])/ 2) & (upper_bounds >= (q[j-1] + q[j])/ 2))
    #             i = j
    #             upper.append(q[i-1])
    #         else:
    #             i += 1
    #     if len(lower) != len(upper):
    #         raise Exception("Length mismatch in lower and upper bounds")
    #     intervals = [[lower[l], upper[l]] for l in range(len(lower))]
    #     return intervals

    
    # def get_disjoint_metrics(self, row):
    #     tv = row['truth']
    #     intvl = row['intervals']
    #     covers = np.any([tv >= i[0] and tv <= i[1] for i in intvl])
    #     widths = np.sum([i[1] - i[0] for i in intvl])
    #     return pd.Series({'covers': covers, 'widths': widths})