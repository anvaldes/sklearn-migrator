from sklearn.cluster import MiniBatchKMeans
from .k_means import _restore_k_means
from ..utils import json_convert

all_features = [
    'batch_size',
    'compute_labels',
    'init',
    'init_size',
    'max_iter',
    'max_no_improvement',
    'n_clusters',
    'n_init',
    'reassignment_ratio',
    'random_state',
    'tol',
    'verbose',
    'cluster_centers_',
    'labels_',
    'inertia_',
    'n_features_in_',
    'n_iter_',
    'feature_names_in_',
    '_n_threads',
    '_algorithm',
    '_tol',
    '_n_init',
    '_n_features_out',
    'n_jobs',
    'precompute_distances'
]

def serialize_mini_batch_kmeans(model: MiniBatchKMeans, version_in: str) -> dict:
    """
    Serialize a fitted MiniBatchKMeans into a JSON-compatible dictionary.

    Parameters
    ----------
    model : MiniBatchKMeans
        A fitted scikit-learn MiniBatchKMeans instance.
    version_in : str
        The sklearn version used to train the model (e.g. '1.2.0').

    Returns
    -------
    dict
        A dictionary containing all necessary data to reconstruct the model.
    """

    metadata = {}

    init_params = model.get_params()

    for p in ['n_jobs', 'precompute_distances', 'algorithm', 'copy_x']:
        init_params.pop(p, None)

    metadata['init_params'] = init_params

    model_dict = model.__dict__

    default_values = {
        'batch_size': getattr(model, 'batch_size', init_params.get('batch_size', 1024)),
        '_n_init': 1,
        '_n_threads': 1,
        '_tol': model.tol,
        '_n_features_out': len(model.cluster_centers_)
            if 'cluster_centers_' in model_dict else None,

        'feature_names_in_': None,
        'n_features_in_': (
            len(model.cluster_centers_[0])
            if 'cluster_centers_' in model_dict
            else None
        ),

        'n_jobs': 1,
        'precompute_distances': 'auto'
    }

    metadata['other_params'] = {af: model_dict.get(af, default_values.get(af, None)) for af in all_features}
    metadata['version_sklearn_in'] = version_in

    return json_convert(metadata)


def deserialize_mini_batch_kmeans(data: dict, version_out: str) -> MiniBatchKMeans:
    """
    Reconstruct a MiniBatchKMeans from a serialized dictionary.

    Parameters
    ----------
    data : dict
        Dictionary produced by serialize_mini_batch_kmeans.
    version_out : str
        The sklearn version of the target environment (e.g. '1.7.0').

    Returns
    -------
    MiniBatchKMeans
        A reconstructed scikit-learn MiniBatchKMeans instance.
    """

    return _restore_k_means(MiniBatchKMeans, data, all_features)
