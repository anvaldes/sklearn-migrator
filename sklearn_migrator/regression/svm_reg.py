
import pandas as pd
import numpy as np
from sklearn.svm import SVR

class Migrated_SVR:

    def __init__(self, metadata):
        
        self.dict = metadata

    def _as_np(self, a):
        return a.to_numpy() if hasattr(a, "to_numpy") else np.asarray(a)
    
    def _kernel_fn(self, X, Y, kind="rbf", gamma=None, coef0=0.0, degree=3):
        X = self._as_np(X).astype(float, copy=False)
        Y = self._as_np(Y).astype(float, copy=False)

        if kind == "linear":
            return X @ Y.T

        if kind == "rbf":
            X2 = np.sum(X * X, axis=1, keepdims=True)
            Y2 = np.sum(Y * Y, axis=1, keepdims=True).T
            return np.exp(-gamma * (X2 + Y2 - 2.0 * (X @ Y.T)))

        if kind == "poly":
            return (gamma * (X @ Y.T) + coef0) ** degree

        if kind == "sigmoid":
            return np.tanh(gamma * (X @ Y.T) + coef0)

        raise ValueError(f"Kernel no soportado: {kind}")

    def predict(self, X):
        
        K = self._kernel_fn(
            X,
            self.dict['support_vectors_'],
            kind = self.dict['params']['kernel'],
            gamma = self.dict['_gamma'],
            coef0 = self.dict['params']['coef0'],
            degree = self.dict['params']['degree']
        )

        alpha = np.ravel(self.dict['dual_coef_'])   # (n_SV,)
        b = float(np.ravel(self.dict['_intercept_'][0]))
    
        return K @ alpha + b

def serialize_svr(model, version_in):

    try:
        prob_A = model._probA.tolist()
    except:
        prob_A = model.probA_.tolist()
    
    try:
        prob_B = model._probB.tolist()
    except:
        prob_B = model.probB_.tolist()
    
    metadata = {
        'meta': 'svr',
        'support_': model.support_.tolist(),
        'n_support_': model.n_support_.tolist(),
        'intercept_': model.intercept_.tolist(),
        'probA': prob_A,
        'probB': prob_B,
        '_intercept_': model._intercept_.tolist(),
        'shape_fit_': model.shape_fit_,
        '_gamma': model._gamma,
        'params': model.get_params(),
        'support_vectors_': model.support_vectors_.tolist(),
        'dual_coef_': model.dual_coef_.tolist(),
        'version_sklearn_in': version_in
    }

    return metadata

def deserialize_svr(data, version_out):

    version_in = data['version_sklearn_in']

    new_model = Migrated_SVR(data)

    return new_model