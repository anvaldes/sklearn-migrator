import pandas as pd
import numpy as np
from sklearn.svm import SVC

class Migrated_SVC:

    def __init__(self, metadata):

        self.dict = metadata
        self._sv = self._as_np(self.dict["support_vectors_"]).astype(float, copy=False)
        self._alpha = np.ravel(self.dict["dual_coef_"]).astype(float, copy=False)
        self._b = float(np.ravel(self.dict["intercept_"])[0])
        self.classes_ = np.asarray(self.dict["classes_"])
        
        if self.classes_.shape[0] != 2:
            raise ValueError("This implementation supports only binary classification (len(classes_)==2).")

        p = self.dict["params"]
        self.kernel = p.get("kernel", "rbf")
        self.gamma = self.dict.get("_gamma", None)
        if self.gamma is None:
            self.gamma = 1.0
        self.coef0 = p.get("coef0", 0.0)
        self.degree = p.get("degree", 3)

        self._has_proba = ("probA" in self.dict) and ("probB" in self.dict) \
                          and (self.dict["probA"] is not None) and (self.dict["probB"] is not None)
        if self._has_proba:
            self.probA_ = np.float64(np.asarray(self.dict["probA"]).item())
            self.probB_ = np.float64(np.asarray(self.dict["probB"]).item())

    def _as_np(self, a):
        return a.to_numpy() if hasattr(a, "to_numpy") else np.asarray(a)

    def _kernel_fn(self, X, Y, kind="rbf", gamma=None, coef0=0.0, degree=3):
        X = self._as_np(X).astype(float, copy=False)
        Y = self._as_np(Y).astype(float, copy=False)

        if kind == "linear":
            return X @ Y.T

        if kind == "rbf":
            if gamma is None:
                raise ValueError("gamma is required for RBF kernel")
            X2 = np.sum(X * X, axis=1, keepdims=True)
            Y2 = np.sum(Y * Y, axis=1, keepdims=True).T
            return np.exp(-gamma * (X2 + Y2 - 2.0 * (X @ Y.T)))

        if kind == "poly":
            if gamma is None:
                raise ValueError("gamma is required for polynomial kernel")
            return (gamma * (X @ Y.T) + coef0) ** degree

        if kind == "sigmoid":
            if gamma is None:
                raise ValueError("gamma is required for sigmoid kernel")
            return np.tanh(gamma * (X @ Y.T) + coef0)

        raise ValueError(f"Unsupported kernel: {kind}")

    def decision_function(self, X):
        K = self._kernel_fn(
            X,
            self._sv,
            kind=self.kernel,
            gamma=self.gamma,
            coef0=self.coef0,
            degree=self.degree,
        )
        f = K @ self._alpha + self._b
        return f

    def predict_proba(self, X):
        if not self._has_proba:
            raise AttributeError("predict_proba unavailable: missing probA/probB (probability=True in the original model).")

        f = self.decision_function(X)
        p0 = 1.0 / (1.0 + np.exp(self.probA_ * (-f) + self.probB_))
        p1 = 1.0 - p0
        probs = np.vstack([p0, p1]).T
        return probs

    def predict(self, X):
        f = self.decision_function(X)
        y = np.where(f > 0, self.classes_[1], self.classes_[0])
        return y

def serialize_svc(model, version_in):
    probA = getattr(model, "probA_", None)
    probB = getattr(model, "probB_", None)
    if probA is not None:
        probA = np.asarray(probA).tolist()
    if probB is not None:
        probB = np.asarray(probB).tolist()

    metadata = {
        "meta": "svc",
        "classes_": getattr(model, "classes_", None).tolist(),
        "support_": model.support_.tolist(),
        "n_support_": model.n_support_.tolist(),
        "support_vectors_": model.support_vectors_.tolist(),
        "dual_coef_": model.dual_coef_.tolist(),
        "intercept_": model.intercept_.tolist(),
        "_intercept_": getattr(model, "_intercept_", model.intercept_).tolist(),
        "shape_fit_": getattr(model, "shape_fit_", None),
        "_gamma": getattr(model, "_gamma", None),
        "params": model.get_params(),
        "probA": probA,
        "probB": probB,
        "version_sklearn_in": version_in,
    }
    return metadata

def deserialize_svc(data, version_out):
    required = ["support_vectors_", "dual_coef_", "intercept_", "classes_", "params"]
    for k in required:
        if k not in data:
            raise KeyError(f"Missing required field in metadata: {k}")

    if "_gamma" in data and data["_gamma"] is not None and isinstance(data["_gamma"], str):
        try:
            data["_gamma"] = float(data["_gamma"])
        except Exception:
            pass

    for k in ("probA", "probB"):
        if k in data and data[k] is not None:
            arr = np.asarray(data[k]).reshape(-1)
            data[k] = [float(arr[0])] if arr.size > 0 else None

    new_model = Migrated_SVC(data)
    return new_model