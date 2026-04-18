import warnings
import numpy as np
from sklearn.linear_model import Ridge

all_features = [ 
    'fit_intercept', 
    'copy_X',
    'n_features_in_',
    'feature_names_in_',
    'tol',
    'n_iter_'
]

def serialize_ridge_reg(model: Ridge, version_in: str) -> dict:
    """
    Serialize a fitted Ridge regression model into a JSON-compatible dictionary.

    Parameters
    ----------
    model : Ridge
        A fitted scikit-learn Ridge instance.
    version_in : str
        The sklearn version used to train the model (e.g. '1.2.0').

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

    model_dict = model.__dict__
    model_dict_keys = list(model_dict.keys())

    default_values = {
        'n_features_in_': len(model.coef_) if model.coef_.ndim == 1 else len(model.coef_[0]),
        'feature_names_in_': None,
        'tol': 1e-6,
        'n_iter_': 1
        }
    
    kdv = list(default_values.keys())

    other_params = {}

    for af in all_features:
        if (af in model_dict_keys) == False:
            other_params[af] = default_values[af]
        else:
            other_params[af] = model_dict[af]

    metadata['other_params'] = other_params

    return metadata


def deserialize_ridge_reg(data: dict, version_out: str) -> Ridge:
    """
    Reconstruct a Ridge regression model from a serialized dictionary.

    Parameters
    ----------
    data : dict
        Dictionary produced by serialize_ridge_reg.
    version_out : str
        The sklearn version of the target environment (e.g. '1.7.0').

    Returns
    -------
    Ridge
        A reconstructed scikit-learn Ridge instance.
    """

    model = Ridge()

    model.alpha = data['alpha']
    model.coef_ = np.array(data['coef_'])
    model.intercept_ = np.array(data['intercept_'])

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