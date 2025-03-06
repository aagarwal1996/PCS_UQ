import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics.pairwise import cdist
from local_conformal_utils import LCP

class Conformal_Localized:
    """
    Conformal Localized Regression
    """
    def __init__ (self, model, alpha = 0.1):
        self.model = clone(model)
        self.alpha = alpha
        self.LCR = None
    
    def fit(self, X, y, alpha = None):
        if alpha is None:
            alpha = self.alpha
        self.model.fit(X, y)
        Vcv = np.abs(y).flatten()
        Dcv = cdist(X, X, metric='euclidean')

        # Set up a grid of localizer bandwidths. (In 1D the code used quantiles of Dcv;
        # here we use the same idea applied to the multivariate distances.)
        max0 = np.max(Dcv) * 2
        min0 = np.quantile(Dcv, 0.01)
        hs = np.exp(np.linspace(np.log(min0), np.log(max0), 20))
        self._calibrate(X, y, Vcv, Dcv, hs, alpha)

    def _calibrate(self, X, y,  Vcv, Dcv, hs, alpha = 0.1):
        eps = np.abs(y - self.model.predict(X)).flatten()
        D = cdist(X, X, metric='euclidean')

        # For the LCP module we need to order the calibration scores.
        order1 = np.argsort(eps) 
        D_ordered = D[order1][:, order1]
        eps_ordered = eps[order1]

        # make lcp
        self.LCR = LCP(H=D_ordered, V=eps_ordered, h=0.2, alpha=alpha, type="distance")

        # Auto-tuning: here we call the auto-tune function using the training data
        auto_ret = self.LCR.LCP_auto_tune(V0=Vcv, H0=Dcv, 
                                          hs=hs, B=2, delta=alpha/2, 
                                          lambda_=1, trace=True)
        self.LCR.h = auto_ret['h']

        # Prepare the calibration/localizer quantities:
        self.LCR.lower_idx()
        self.LCR.cumsum_unnormalized()
        self.calibrated = True

       
    def predict(self, x_test):
        if self.calibrated == False:
            raise ValueError("Model must be calibrated before calling predict.")
        # do test prediction
        y_test_pred = self.model.predict(x_test)

        # computer distances
        Dnew = cdist(x_test, self.x_val, metric='euclidean')
        DnewT = Dnew.T

        # make predictions
        self.LCR.LCP_construction(Hnew=Dnew[:, self.__order1], HnewT=DnewT[self.__order1, :])
        deltaLCP = self.LCR.band_V
        lb = y_test_pred - deltaLCP
        ub = y_test_pred + deltaLCP

        self.prediction_intervals = pd.DataFrame({"lb": lb, "ub": ub, "point_est": y_test_pred})
        self.predicted = True
        return self.prediction_intervals
    
   