import warnings
import numpy as np
from sklearn.cluster import KMeans

all_features = [
    'algorithm',
    'cluster_centers_',
    'copy_x', 
    'inertia_',
    'init',
    'labels_',
    'max_iter',
    'n_clusters',
    'n_init', 
    'n_iter_',
    'tol',
    'verbose',
    '_algorithm',
    '_n_features_out', 
    '_n_init',
    '_n_threads',
    '_tol', 
    'feature_names_in_', 
    'n_features_in_', 
    'n_jobs',
    'precompute_distances'
]

def serialize_k_means(model: KMeans, version_in: str) -> dict:
    """
    Serialize a fitted KMeans into a JSON-compatible dictionary.

    Parameters
    ----------
    model : KMeans
        A fitted scikit-learn KMeans instance.
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
        del init_params['n_jobs']
    except (KeyError, AttributeError):
        pass
    except Exception as e:
        warnings.warn(f"Could not delete field 'n_jobs': {type(e).__name__}: {e}")

    try:
        del init_params['precompute_distances']
    except (KeyError, AttributeError):
        pass
    except Exception as e:
        warnings.warn(f"Could not delete field 'precompute_distances': {type(e).__name__}: {e}")

    metadata['init_params'] = init_params

    model_dict = model.__dict__
    model_dict_keys = list(model_dict.keys())

    default_values = {
        '_algorithm': 'lloyd',
        '_n_features_out': len(model.cluster_centers_),
        '_n_init': 1,
        '_n_threads': 1,
        '_tol': model.tol,
        'feature_names_in_': None,
        'n_features_in_': len(model.cluster_centers_[0]),
        'n_jobs': 1,
        'precompute_distances': 'auto'
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

def deserialize_k_means(data: dict, version_out: str) -> KMeans:
    """
    Reconstruct a KMeans from a serialized dictionary.

    Parameters
    ----------
    data : dict
        Dictionary produced by serialize_k_means.
    version_out : str
        The sklearn version of the target environment (e.g. '1.7.0').

    Returns
    -------
    KMeans
        A reconstructed scikit-learn KMeans instance.
    """
    
    version_in = data['version_sklearn_in']
    init_params = data['init_params']

    new_model = KMeans(**init_params)

    array_fields = [
        'cluster_centers_',
        'labels_',
        'feature_names_in_'
    ]

    other_params = data['other_params']

    for af in all_features:
        if af not in other_params:
            continue

        value = other_params[af]

        if af in array_fields and value is not None and not isinstance(value, np.ndarray):
            value = np.array(value)

        new_model.__dict__[af] = value

    return new_model