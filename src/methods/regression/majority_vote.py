
import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.base import clone
from split_conformal import SplitConformal
from sklearn.model_selection import train_test_split

class Conformal_Majority_Vote:
    def __init__ (self, models, alpha = 0.1, seed = 42):
        """
        Initialize the Conformal Majority Vote model

        Args:
            models: dictionary of model names and models
        """
        self.models = {model_name: clone(model) for model_name, model in models.items()}
        self.K = len(models)
        self.alpha = alpha
        self.seed = seed

    def fit(self, X, y, alpha = None):
        if alpha is None:
            alpha = self.alpha
        self.conformals = {}
        X_train, X_calib, y_train, y_calib = train_test_split(X, y, test_size=0.5, random_state=self.seed)
        for model_name, model in self.models.items():
            split_conformal_model = SplitConformal(model, alpha)
            split_conformal_model.fit(X_train, y_train)
            self.conformals[model_name] = split_conformal_model
        self.calibrate(X_calib, y_calib)
    
    def calibrate(self, X, y, alpha=0.1):
        if self.trained == False:
            raise ValueError("Model must be trained before calling calibrate.")
        qs = {}
        for model_name, conformal in self.conformals.items():
            q = conformal.calibrate(X, y, alpha/2)
            qs[model_name] = q

        self.calibrated = True
        return qs
    
    def predict(self, x_test, tau = 0.5):
        if self.calibrated == False:
            raise ValueError("Model must be calibrated before calling predict.")
        self.tau = tau
        lb_cols = [f'{model_name}_lb' for model_name in self.conformals.keys()]
        ub_cols = [f'{model_name}_ub' for model_name in self.conformals.keys()]
        all_conformals = pd.DataFrame(np.zeros((x_test.shape[0], 2*self.K)), 
                                      columns=lb_cols + ub_cols)
        for model_name, conformal in self.conformals.items():
            single_conformal_pred = conformal.predict(x_test)
            all_conformals[f'{model_name}_lb'] = single_conformal_pred['lb']
            all_conformals[f'{model_name}_ub'] = single_conformal_pred['ub']
        self.all_conformals = all_conformals
        self.prediction_intervals = pd.DataFrame(all_conformals.apply(lambda row: self.get_majority_vote(row, self.K, self.tau),axis=1),
                                                columns=['intervals'])

        self.predicted = True
        return self.prediction_intervals

    def get_majority_vote(self, row, K, tau):
        lower_bounds = row.iloc[:K].to_numpy()
        upper_bounds = row.iloc[K:].to_numpy()
        q = np.sort(row)
        i = 1
        lower = []
        upper = []
        while i < 2*K:
            cond_i = np.mean((lower_bounds <= (q[i-1] + q[i])/ 2) & (upper_bounds >= (q[i-1] + q[i])/ 2))
            if i == 10 and row.name == 1:
                print(lower_bounds, upper_bounds, q[i-1], q[i], (lower_bounds <= (q[i-1] + q[i])/ 2) & (upper_bounds >= (q[i-1] + q[i])/ 2))
            if cond_i > tau:
                lower.append(q[i-1])
                j = i
                cond_j = np.mean((lower_bounds <= (q[j-1] + q[j])/ 2) & (upper_bounds >= (q[j-1] + q[j])/ 2))
                while (cond_j > tau) and (j < 2*K):
                    j += 1
                    cond_j = np.mean((lower_bounds <= (q[j-1] + q[j])/ 2) & (upper_bounds >= (q[j-1] + q[j])/ 2))
                i = j
                upper.append(q[i-1])
            else:
                i += 1
        if len(lower) != len(upper):
            raise Exception("Length mismatch in lower and upper bounds")
        intervals = [[lower[l], upper[l]] for l in range(len(lower))]
        return intervals

    def evaluate(self, y_test, scale_width=True):
        """
        Evaluates the prediction intervals 
        and computes coverage and length metrics.

        Parameters
        ----------
        y_test : np.ndarray
            The true target values for the test set.

        Returns
        -------
        pd.DataFrame
            DataFrame containing evaluation metrics for each alpha.
        """
        if self.predicted is False:
            raise ValueError("Please call 'predict' first to generate prediction intervals.")
        
        intervals = deepcopy(self.prediction_intervals)
        intervals['truth'] = y_test
        disjoint_metrics = intervals.apply(lambda r: self.get_disjoint_metrics(r), axis=1)

        coverage = disjoint_metrics['covers'].mean()
        avg_length = disjoint_metrics['widths'].mean()
        med_length = np.median(disjoint_metrics['widths'])
        range_y_test = y_test.max() - y_test.min()

        # Compile results into a DataFrame
        results_df = pd.DataFrame([{
            "coverage": coverage,
            "avg_length": avg_length,
            "median_length": med_length,
            "range_y_test": range_y_test,
            "alpha": self.alpha
        }])

        if scale_width:
            results_df["scaled_avg_length"] = avg_length / range_y_test
            results_df["scaled_median_length"] = med_length / range_y_test

        return results_df
    
    def get_disjoint_metrics(self, row):
        tv = row['truth']
        intvl = row['intervals']
        covers = np.any([tv >= i[0] and tv <= i[1] for i in intvl])
        widths = np.sum([i[1] - i[0] for i in intvl])
        return pd.Series({'covers': covers, 'widths': widths})
    
    def evaluate_subgroups(self, y_test, subgroups, scale_width=True):
        if self.predicted == False:
            raise ValueError("Please call 'predict' first to generate prediction intervals.")
        pred_modified = deepcopy(self.prediction_intervals)
        pred_modified["truth"] = y_test
        pred_modified = pred_modified.apply(lambda r: self.get_disjoint_metrics(r), axis=1)
        pred_modified['truth'] = y_test
        pred_modified["subgroup"] = subgroups

        group_metrics = pred_modified.groupby("subgroup").agg(
            coverage = ("covers", "mean"),
            avg_length = ("widths", "mean"),
            median_length = ("widths", "median"),
            range_y_test = ("truth", lambda x: x.max() - x.min())
        ).reset_index()
        group_metrics["alpha"] = self.alpha

        if scale_width:
            range_y_test = y_test.max() - y_test.min()
            group_metrics["scaled_avg_length"] = group_metrics["avg_length"] / range_y_test
            group_metrics["scaled_median_length"] = group_metrics["median_length"] / range_y_test

        return group_metrics
    