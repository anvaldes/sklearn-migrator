import warnings
import numpy as np
from sklearn.decomposition import PCA

all_features = [
    '_fit_svd_solver',
    'components_',
    'copy',
    'explained_variance_',
    'explained_variance_ratio_',
    'iterated_power',
    'mean_',
    'n_components',
    'n_components_',
    'n_samples_',
    'noise_variance_',
    'singular_values_',
    'svd_solver',
    'tol',
    'whiten',
    'feature_names_in_',
    'n_features_',
    'n_features_in_',
    'power_iteration_normalizer'
    ]


def serialize_pca(model: PCA, version_in: str) -> dict:
    """
    Serialize a fitted PCA into a JSON-compatible dictionary.

    Parameters
    ----------
    model : PCA
        A fitted scikit-learn PCA instance.
    version_in : str
        The sklearn version used to train the model (e.g. '1.2.0').

    Returns
    -------
    dict
        A dictionary containing all necessary data to reconstruct the model.
    """

    metadata = {}

    init_params = model.get_params()

    try:
        del init_params['n_oversamples']
    except (KeyError, AttributeError):
        pass
    except Exception as e:
        warnings.warn(f"Could not delete field 'n_oversamples': {type(e).__name__}: {e}")

    try:
        del init_params['power_iteration_normalizer']
    except (KeyError, AttributeError):
        pass
    except Exception as e:
        warnings.warn(f"Could not delete field 'power_iteration_normalizer': {type(e).__name__}: {e}")

    metadata['init_params'] = init_params

    model_dict = model.__dict__
    model_dict_keys = list(model_dict.keys())

    default_values = {
        'feature_names_in_': None,
        'n_features_': len(model_dict['mean_']),
        'n_features_in_': len(model_dict['mean_']),
        'power_iteration_normalizer': 'auto'
        }

    other_params = {}
    
    for af in all_features:
        if (af in model_dict_keys) == False:
            other_params[af] = default_values[af]
        else:
            other_params[af] = model_dict[af]

    metadata['other_params'] = other_params
    metadata['version_sklearn_in'] = version_in

    return metadata


def deserialize_pca(data: dict, version_out: str) -> PCA:
    """
    Reconstruct a PCA from a serialized dictionary.

    Parameters
    ----------
    data : dict
        Dictionary produced by serialize_pca.
    version_out : str
        The sklearn version of the target environment (e.g. '1.7.0').

    Returns
    -------
    PCA
        A reconstructed scikit-learn PCA instance.
    """
    
    version_in = data['version_sklearn_in']
    init_params = data['init_params']

    new_model = PCA(**init_params)

    array_fields = [
        'components_',
        'explained_variance_',
        'explained_variance_ratio_',
        'mean_',
        'singular_values_'
    ]

    other_params = data['other_params']

    for af in all_features:
        if af not in other_params:
            continue  # field not present in this sklearn version
        value = other_params[af]
        if af in array_fields and value is not None and not isinstance(value, np.ndarray):
            value = np.array(value)
        try:
            new_model.__dict__[af] = value
        except AttributeError:
            pass  # attribute not settable in this sklearn version
        except Exception as e:
            warnings.warn(
                f"Could not set field '{af}' on {type(new_model).__name__}: "
                f"{type(e).__name__}: {e}. Field will be skipped.",
                UserWarning,
            )

    return new_model