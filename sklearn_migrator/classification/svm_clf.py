import pandas as pd
import numpy as np
from sklearn.svm import SVC


class Migrated_SVC:
    """
    A pure-NumPy reimplementation of a binary SVC for cross-version compatibility.

    This class reconstructs the prediction behaviour of a fitted scikit-learn
    SVC without relying on the original sklearn object, making it safe to use
    across different sklearn versions.

    Parameters
    ----------
    metadata : dict
        Dictionary produced by serialize_svc.
    """

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

    def _as_np(self, a) -> np.ndarray:
        """
        Convert a pandas DataFrame or array-like object to a numpy array.

        Parameters
        ----------
        a : array-like or pd.DataFrame
            Input data.

        Returns
        -------
        np.ndarray
            Numpy array representation of the input.
        """

        return a.to_numpy() if hasattr(a, "to_numpy") else np.asarray(a)

    def _kernel_fn(self, X, Y, kind: str = "rbf", gamma: float = None, coef0: float = 0.0, degree: int = 3) -> np.ndarray:
        """
        Compute the kernel matrix between X and Y.

        Parameters
        ----------
        X : array-like
            First input matrix of shape (n_samples_X, n_features).
        Y : array-like
            Second input matrix of shape (n_samples_Y, n_features).
        kind : str
            Kernel type: 'linear', 'rbf', 'poly', or 'sigmoid'.
        gamma : float, optional
            Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
        coef0 : float
            Independent term for 'poly' and 'sigmoid' kernels.
        degree : int
            Degree for the polynomial kernel.

        Returns
        -------
        np.ndarray
            Kernel matrix of shape (n_samples_X, n_samples_Y).
        """

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

    def decision_function(self, X) -> np.ndarray:
        """
        Compute the decision function for the input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        np.ndarray
            Decision function values of shape (n_samples,).
        """

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
    
    def _sigmoid_predict(self, f: np.ndarray) -> np.ndarray:
        """
        Apply Platt scaling to decision function values using a numerically
        stable two-branch sigmoid, mirroring libsvm's sigmoid_predict().

        For fApB >= 0 uses exp(-fApB) / (1 + exp(-fApB)) and for
        fApB < 0 uses 1 / (1 + exp(fApB)), avoiding overflow in both cases.

        Parameters
        ----------
        f : np.ndarray of shape (n_samples,)
            Decision function values from decision_function().

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Probability estimates for the positive class (classes_[1]).
        """

        fApB = self.probA_ * f + self.probB_
        pos = fApB >= 0
        result = np.empty_like(fApB)
        result[pos]  = np.exp(-fApB[pos])  / (1.0 + np.exp(-fApB[pos]))
        result[~pos] = 1.0                 / (1.0 + np.exp( fApB[~pos]))
        return result

    def predict_proba(self, X) -> np.ndarray:
        """
        Compute class probabilities using Platt scaling.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        np.ndarray
            Class probabilities of shape (n_samples, 2).

        Raises
        ------
        AttributeError
            If the original model was not trained with probability=True.
        """

        if not self._has_proba:
            raise AttributeError("predict_proba unavailable: missing probA/probB (probability=True in the original model).")

        f = self.decision_function(X)
        p1 = self._sigmoid_predict(f)
        p0 = 1.0 - p1
        probs = np.vstack([p0, p1]).T
        return probs

    def predict(self, X) -> np.ndarray:
        """
        Predict class labels for the input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        np.ndarray
            Predicted class labels of shape (n_samples,).
        """

        f = self.decision_function(X)
        y = np.where(f > 0, self.classes_[1], self.classes_[0])
        return y

def serialize_svc(model: SVC, version_in: str) -> dict:
    """
    Serialize a fitted SVC into a JSON-compatible dictionary.

    Parameters
    ----------
    model : SVC
        A fitted scikit-learn SVC instance.
    version_in : str
        The sklearn version used to train the model (e.g. '1.2.0').

    Returns
    -------
    dict
        A dictionary containing all necessary data to reconstruct the model.
    """

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

def deserialize_svc(data: dict, version_out: str) -> Migrated_SVC:
    """
    Reconstruct a Migrated_SVC from a serialized dictionary.

    Parameters
    ----------
    data : dict
        Dictionary produced by serialize_svc.
    version_out : str
        The sklearn version of the target environment (e.g. '1.7.0').

    Returns
    -------
    Migrated_SVC
        A reconstructed Migrated_SVC instance compatible with the target environment.
    """

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