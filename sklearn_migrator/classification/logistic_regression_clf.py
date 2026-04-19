import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression

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

    model_dict = model.__dict__
    model_dict_keys = list(model_dict.keys())

    default_values = {
        'n_features_in_': len(model.coef_.tolist()[0]),
        'feature_names_in_': None,
        'multi_class': model.get_params().get('multi_class', None)
    }

    kdv = list(default_values.keys())

    other_params = {}

    for af in all_features:
        if (af in model_dict_keys) == False:
            other_params[af] = default_values[af]
        else:
            other_params[af] = model_dict[af]

    if other_params['multi_class'] == 'deprecated':
        del other_params['multi_class']

    metadata['other_params'] = other_params

    return metadata

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

    for af in all_features:
        try:
            model.__dict__[af] = data['other_params'][af]
        except KeyError:
            pass  # field not present in this sklearn version
        except AttributeError:
            pass  # attribute not settable in this sklearn version
        except Exception as e:
            warnings.warn(
                f"Could not set field '{af}' on {type(model).__name__}: "
                f"{type(e).__name__}: {e}. Field will be skipped.",
                UserWarning,
            )

    return model