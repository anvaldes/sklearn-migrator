import numpy as np
from sklearn.linear_model import LogisticRegression
from ..utils import json_convert, collect_other_params, restore_other_params

all_features = [
    'warm_start',
    'penalty',
    'dual',
    'class_weight',
    'n_jobs',
    'max_iter',
    'fit_intercept',
    'intercept_scaling',
    'multi_class',
    'solver',
    'verbose',
    'C',
    'l1_ratio',
    'tol',
    'n_features_in_',
    'feature_names_in_'
]

def serialize_logistic_regression_clf(model: LogisticRegression, version_in: str) -> dict:
    """
    Serialize a fitted LogisticRegression into a JSON-compatible dictionary.

    Parameters
    ----------
    model : LogisticRegression
        A fitted scikit-learn LogisticRegression instance.
    version_in : str
        The sklearn version used to train the model (e.g. '1.2.0').

    Returns
    -------
    dict
        A dictionary containing all necessary data to reconstruct the model.
    """

    metadata = {
        'classes_': model.classes_.tolist(),
        'coef_': model.coef_.tolist(),
        'intercept_': model.intercept_.tolist(),
        'n_iter_': model.n_iter_.tolist(),
        'params': model.get_params(),
        'version_sklearn_in': version_in
    }

    default_values = {
        'n_features_in_': len(model.coef_.tolist()[0]),
        'feature_names_in_': None,
        'multi_class': model.get_params().get('multi_class', None)
    }

    other_params = collect_other_params(model, all_features, default_values)

    if other_params['multi_class'] == 'deprecated':
        del other_params['multi_class']

    metadata['other_params'] = other_params

    return json_convert(metadata)

def deserialize_logistic_regression_clf(data: dict, version_out: str) -> LogisticRegression:
    """
    Reconstruct a LogisticRegression from a serialized dictionary.

    Parameters
    ----------
    data : dict
        Dictionary produced by serialize_logistic_regression_clf.
    version_out : str
        The sklearn version of the target environment (e.g. '1.7.0').

    Returns
    -------
    LogisticRegression
        A reconstructed scikit-learn LogisticRegression instance.
    """

    model = LogisticRegression(data['params'])

    model.classes_ = np.array(data['classes_'])
    model.coef_ = np.array(data['coef_'])
    model.intercept_ = np.array(data['intercept_'])
    model.n_iter_ = np.array(data['n_iter_'])

    restore_other_params(model, all_features, data)

    return model