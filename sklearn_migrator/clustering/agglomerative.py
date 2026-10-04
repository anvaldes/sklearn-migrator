import numpy as np
from sklearn.cluster import AgglomerativeClustering
from ..utils import json_convert, version_tuple

all_features = [
    'n_features_in_',
    'labels_',
    'n_connected_components_',
    'children_',
    'distances_',
    'feature_names_in_',
]


def serialize_agglomerative(model: AgglomerativeClustering, version_in: str) -> dict:
    """
    Serialize a fitted AgglomerativeClustering into a JSON-compatible dictionary.

    Parameters
    ----------
    model : AgglomerativeClustering
        A fitted scikit-learn AgglomerativeClustering instance.
    version_in : str
        The sklearn version used to train the model (e.g. '1.2.0').

    Returns
    -------
    dict
        A dictionary containing all necessary data to reconstruct the model.
    """

    metadata = {}

    init_params = model.get_params()

    if 'metric' not in init_params and 'affinity' in init_params:
        init_params['metric'] = init_params['affinity']

    init_params.pop('affinity', None)

    metadata['init_params'] = init_params

    model_dict = model.__dict__

    metadata['other_params'] = {af: model_dict.get(af, None) for af in all_features}
    metadata['version_sklearn_in'] = version_in

    return json_convert(metadata)


def deserialize_agglomerative(data: dict, version_out: str) -> AgglomerativeClustering:
    """
    Reconstruct an AgglomerativeClustering from a serialized dictionary.

    Parameters
    ----------
    data : dict
        Dictionary produced by serialize_agglomerative.
    version_out : str
        The sklearn version of the target environment (e.g. '1.7.0').

    Returns
    -------
    AgglomerativeClustering
        A reconstructed scikit-learn AgglomerativeClustering instance.
    """

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
        if value is not None and af in array_fields and not isinstance(value, np.ndarray):
            value = np.array(value)

        new_model.__dict__[af] = value

    return new_model
