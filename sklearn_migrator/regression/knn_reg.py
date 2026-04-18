import numpy as np
from sklearn.neighbors import KNeighborsRegressor

all_features = [
    "_fit_X",
    "_y",
    "feature_names_in_",
]

def serialize_knn_reg(model: KNeighborsRegressor, version_in: str) -> dict:
    """
    Serialize a fitted KNeighborsRegressor into a JSON-compatible dictionary.

    Parameters
    ----------
    model : KNeighborsRegressor
        A fitted scikit-learn KNeighborsRegressor instance.
    version_in : str
        The sklearn version used to train the model (e.g. '1.2.0').

    Returns
    -------
    dict
        A dictionary containing all necessary data to reconstruct the model.
    """

    metadata = {}

    init_params = model.get_params()
    metadata["init_params"] = init_params

    model_dict = model.__dict__
    model_dict_keys = list(model_dict.keys())

    default_values = {
        "feature_names_in_": None,
    }

    other_params = {}

    for af in all_features:
        if af in model_dict_keys:
            val = model_dict[af]
        else:
            val = default_values.get(af, None)

        if isinstance(val, np.ndarray):
            val = val.tolist()

        other_params[af] = val

    metadata["other_params"] = other_params
    metadata["version_sklearn_in"] = version_in

    return metadata


def deserialize_knn_reg(data: dict, version_out: str) -> KNeighborsRegressor:
    """
    Reconstruct a KNeighborsRegressor from a serialized dictionary.

    Parameters
    ----------
    data : dict
        Dictionary produced by serialize_knn_reg.
    version_out : str
        The sklearn version of the target environment (e.g. '1.7.0').

    Returns
    -------
    KNeighborsRegressor
        A reconstructed scikit-learn KNeighborsRegressor instance.
    """

    init_params = data["init_params"]
    other_params = data["other_params"]

    X = other_params["_fit_X"]
    y = other_params["_y"]

    X = np.asarray(X)

    if hasattr(y, "values"):
        y = y.values

    y = np.asarray(y)

    new_model = KNeighborsRegressor(**init_params)
    new_model.fit(X, y)

    if "feature_names_in_" in other_params and other_params["feature_names_in_"] is not None:
        new_model.feature_names_in_ = np.array(other_params["feature_names_in_"])

    return new_model