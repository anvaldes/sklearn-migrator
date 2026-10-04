import numpy as np
from sklearn.cluster import KMeans
from ..utils import json_convert, collect_other_params

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

    for p in ['n_jobs', 'precompute_distances']:
        init_params.pop(p, None)

    metadata['init_params'] = init_params

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

    metadata['other_params'] = collect_other_params(model, all_features, default_values)
    metadata['version_sklearn_in'] = version_in

    return json_convert(metadata)

def _restore_k_means(model_class, data: dict, all_features: list):
    """
    Reconstruct a KMeans-like model from a serialized dictionary.

    Parameters
    ----------
    model_class : type
        KMeans or MiniBatchKMeans.
    data : dict
        Dictionary produced by the serializer of model_class.
    all_features : list
        Names of the fields to set on the model.

    Returns
    -------
    KMeans or MiniBatchKMeans
        A reconstructed scikit-learn instance of model_class.
    """

    new_model = model_class(**data['init_params'])

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

    return _restore_k_means(KMeans, data, all_features)
