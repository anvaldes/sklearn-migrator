import warnings
import numpy as np
from sklearn.linear_model import LinearRegression

all_features = [ 
    'fit_intercept', 
    'n_jobs', 
    'copy_X',
    'normalize',
    'n_features_in_',
    'positive',
    'feature_names_in_',
    'tol'
]

def serialize_linear_regression_reg(model: LinearRegression, version_in: str) -> dict:
    """
    Serialize a fitted LinearRegression into a JSON-compatible dictionary.

    Parameters
    ----------
    model : LinearRegression
        A fitted scikit-learn LinearRegression instance.
    version_in : str
        The sklearn version used to train the model (e.g. '1.2.0').

    Returns
    -------
    dict
        A dictionary containing all necessary data to reconstruct the model.
    """

    metadata = {
        'coef_': model.coef_.tolist(),
        'singular_': model.singular_.tolist(),
        'intercept_': model.intercept_.tolist(),
        'rank_': model.rank_,
        'version_sklearn_in': version_in
    }

    model_dict = model.__dict__
    model_dict_keys = list(model_dict.keys())

    default_values = {
        'normalize': False,
        'n_features_in_': len(model.coef_) if model.coef_.ndim == 1 else len(model.coef_[0]),
        'positive': False,
        'feature_names_in_': None,
        'tol': 1e-6
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


def deserialize_linear_regression_reg(data: dict, version_out: str) -> LinearRegression:
    """
    Reconstruct a LinearRegression from a serialized dictionary.

    Parameters
    ----------
    data : dict
        Dictionary produced by serialize_linear_regression_reg.
    version_out : str
        The sklearn version of the target environment (e.g. '1.7.0').

    Returns
    -------
    LinearRegression
        A reconstructed scikit-learn LinearRegression instance.
    """

    model = LinearRegression()

    model.coef_ = np.array(data['coef_'])
    model.singular_ = np.array(data['singular_'])
    model.intercept_ = np.array(data['intercept_'])
    model.rank_ = data['rank_']

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