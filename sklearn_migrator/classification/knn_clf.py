import numpy as np
from sklearn.neighbors import KNeighborsClassifier

all_features = [
    "_fit_X",
    "_y",
    "classes_",
    "feature_names_in_",
]

def serialize_knn_clf(model, version_in):

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


def deserialize_knn_clf(data, version_out):

    init_params = data["init_params"]
    other_params = data["other_params"]

    X = other_params["_fit_X"]
    y = other_params["_y"]

    X = np.asarray(X)

    if hasattr(y, "values"):
        y = y.values
    y = np.asarray(y).ravel()

    new_model = KNeighborsClassifier(**init_params)
    new_model.fit(X, y)

    if "feature_names_in_" in other_params and other_params["feature_names_in_"] is not None:
        new_model.feature_names_in_ = np.array(other_params["feature_names_in_"])

    return new_model
