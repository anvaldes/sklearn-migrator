import numpy as np
from sklearn.cluster import MiniBatchKMeans

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

def serialize_mini_batch_kmeans(model, version_in):

    metadata = {}

    init_params = model.get_params()

    for p in ['n_jobs', 'precompute_distances', 'algorithm', 'copy_x']:
        try:
            del init_params[p]
        except:
            pass

    metadata['init_params'] = init_params

    model_dict = model.__dict__
    model_dict_keys = list(model_dict.keys())

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

    other_params = {}

    for af in all_features:
        if af in model_dict_keys:
            other_params[af] = model_dict[af]
        else:
            other_params[af] = default_values.get(af, None)

    metadata['other_params'] = other_params
    metadata['version_sklearn_in'] = version_in

    return metadata


def deserialize_mini_batch_kmeans(data, version_out):

    version_in = data['version_sklearn_in']
    init_params = data['init_params']

    new_model = MiniBatchKMeans(**init_params)

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
