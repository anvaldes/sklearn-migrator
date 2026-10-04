import numpy as np
from ..utils import json_convert, collect_other_params, restore_other_params

all_features = [
    'fit_intercept',
    'copy_X',
    'n_features_in_',
    'feature_names_in_',
    'tol',
    'n_iter_'
]


def _serialize_penalized_linear(model, version_in: str) -> dict:
    """
    Serialize a fitted Lasso or Ridge model into a JSON-compatible dictionary.

    Parameters
    ----------
    model : Lasso or Ridge
        A fitted scikit-learn Lasso or Ridge instance.
    version_in : str
        The sklearn version used to train the model.

    Returns
    -------
    dict
        A dictionary containing all necessary data to reconstruct the model.
    """

    metadata = {
        'alpha': model.alpha,
        'coef_': model.coef_.tolist(),
        'intercept_': model.intercept_.tolist(),
        'version_sklearn_in': version_in
    }

    default_values = {
        'n_features_in_': len(model.coef_) if model.coef_.ndim == 1 else len(model.coef_[0]),
        'feature_names_in_': None,
        'tol': 1e-6,
        'n_iter_': 1
        }

    metadata['other_params'] = collect_other_params(model, all_features, default_values)

    return json_convert(metadata)


def _deserialize_penalized_linear(model_class, data: dict):
    """
    Reconstruct a Lasso or Ridge model from a serialized dictionary.

    Parameters
    ----------
    model_class : type
        Lasso or Ridge.
    data : dict
        Dictionary produced by _serialize_penalized_linear.

    Returns
    -------
    Lasso or Ridge
        A reconstructed scikit-learn Lasso or Ridge instance.
    """

    model = model_class()

    model.alpha = data['alpha']
    model.coef_ = np.array(data['coef_'])
    model.intercept_ = np.array(data['intercept_'])

    restore_other_params(model, all_features, data)

    return model
