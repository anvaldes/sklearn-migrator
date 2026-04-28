import warnings
import pandas as pd
import numpy as np
from sklearn.svm import SVR

class Migrated_SVR:
    """
    A pure-NumPy reimplementation of SVR for cross-version compatibility.

    This class reconstructs the prediction behaviour of a fitted scikit-learn
    SVR without relying on the original sklearn object, making it safe to use
    across different sklearn versions.

    Parameters
    ----------
    metadata : dict
        Dictionary produced by serialize_svr.
    """

    def __init__(self, metadata):
        
        self.dict = metadata
    
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
            X2 = np.sum(X * X, axis=1, keepdims=True)
            Y2 = np.sum(Y * Y, axis=1, keepdims=True).T
            return np.exp(-gamma * (X2 + Y2 - 2.0 * (X @ Y.T)))

        if kind == "poly":
            return (gamma * (X @ Y.T) + coef0) ** degree

        if kind == "sigmoid":
            return np.tanh(gamma * (X @ Y.T) + coef0)

        raise ValueError(f"Kernel no soportado: {kind}")

    def predict(self, X) -> np.ndarray:
        """
        Predict target values for the input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        np.ndarray
            Predicted target values of shape (n_samples,).
        """
        
        K = self._kernel_fn(
            X,
            self.dict['support_vectors_'],
            kind = self.dict['params']['kernel'],
            gamma = self.dict['_gamma'],
            coef0 = self.dict['params']['coef0'],
            degree = self.dict['params']['degree']
        )

        alpha = np.asarray(self.dict["dual_coef_"], dtype=float).ravel()

        b_arr = np.asarray(self.dict["_intercept_"], dtype=float).ravel()
        b = float(b_arr[0])

        return K @ alpha + b

def serialize_svr(model: SVR, version_in: str) -> dict:
    """
    Serialize a fitted SVR into a JSON-compatible dictionary.

    Parameters
    ----------
    model : SVR
        A fitted scikit-learn SVR instance.
    version_in : str
        The sklearn version used to train the model (e.g. '1.2.0').

    Returns
    -------
    dict
        A dictionary containing all necessary data to reconstruct the model.
    """

    try:
        prob_A = model._probA.tolist()
    except (KeyError, AttributeError):
        prob_A = model.probA_.tolist()
    except Exception as e:
        warnings.warn(f"Could not get field '_probA': {type(e).__name__}: {e}")
        prob_A = None

    try:
        prob_B = model._probB.tolist()
    except (KeyError, AttributeError):
        prob_B = model.probB_.tolist()
    except Exception as e:
        warnings.warn(f"Could not get field '_probB': {type(e).__name__}: {e}")
        prob_B = None
    
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

def deserialize_svr(data: dict, version_out: str) -> Migrated_SVR:
    """
    Reconstruct a Migrated_SVR from a serialized dictionary.

    Parameters
    ----------
    data : dict
        Dictionary produced by serialize_svr.
    version_out : str
        The sklearn version of the target environment (e.g. '1.7.0').

    Returns
    -------
    Migrated_SVR
        A reconstructed Migrated_SVR instance compatible with the target environment.
    """

    version_in = data['version_sklearn_in']

    new_model = Migrated_SVR(data)

    return new_model