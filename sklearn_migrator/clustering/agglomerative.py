import numpy as np
from sklearn.cluster import AgglomerativeClustering

all_features = [
    'n_features_in_',
    'labels_',
    'n_connected_components_',
    'children_',
    'distances_',
    'feature_names_in_',
]


def version_tuple(version):
    parts = version.split(".")
    parts = (parts + ["0", "0"])[:3]
    return tuple(int(p) for p in parts)


def serialize_agglomerative(model, version_in):

    metadata = {}

    init_params = model.get_params()

    if 'metric' not in init_params and 'affinity' in init_params:
        init_params['metric'] = init_params['affinity']

    try:
        del init_params['affinity']
    except KeyError:
        pass

    metadata['init_params'] = init_params

    model_dict = model.__dict__
    model_keys = list(model_dict.keys())

    default_values = {
        'n_features_in_': None,
        'labels_': None,
        'n_connected_components_': None,
        'children_': None,
        'distances_': None,
        'feature_names_in_': None,
    }

    other_params = {}

    for af in all_features:
        if af in model_keys:
            val = model_dict[af]
        else:
            val = default_values.get(af, None)

        if isinstance(val, np.ndarray):
            val = val.tolist()

        other_params[af] = val

    metadata['other_params'] = other_params
    metadata['version_sklearn_in'] = version_in

    return metadata


def deserialize_agglomerative(data, version_out):

    init_params = data['init_params'].copy()
    other_params = data['other_params']

    v_out = version_tuple(version_out)

    if v_out >= (1, 2, 0) and init_params.get("metric") is None:
        init_params["metric"] = "euclidean"

    if v_out < (1, 2, 0):
        if 'metric' in init_params:
            if 'affinity' not in init_params:
                init_params['affinity'] = init_params['metric']
            del init_params['metric']

        if init_params.get("linkage") == "ward" and init_params.get("affinity") is None:
            init_params["affinity"] = "euclidean"

    init_params.pop("pooling_func", None)
    init_params.pop("compute_distances", None)

    new_model = AgglomerativeClustering(**init_params)

    if v_out < (1, 2, 0) and hasattr(new_model, "metric") and new_model.metric is None:
        aff = getattr(new_model, "affinity", None)
        if isinstance(aff, str):
            new_model.metric = aff
        elif getattr(new_model, "linkage", None) == "ward":
            new_model.metric = "euclidean"

    array_fields = [
        'children_',
        'distances_',
    ]

    for af, value in other_params.items():
        if value is None:
            new_model.__dict__[af] = None
            continue

        if af in array_fields and not isinstance(value, np.ndarray):
            value = np.array(value)

        new_model.__dict__[af] = value

    return new_model
