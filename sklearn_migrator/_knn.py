import numpy as np
from .utils import json_convert


def _serialize_knn(model, version_in: str, all_features: list) -> dict:
    """
    Serialize a fitted k-nearest neighbors model into a JSON-compatible dictionary.

    Parameters
    ----------
    model : KNeighborsRegressor or KNeighborsClassifier
        A fitted scikit-learn k-nearest neighbors instance.
    version_in : str
        The sklearn version used to train the model.
    all_features : list
        Names of the fields to read from the model.

    Returns
    -------
    dict
        A dictionary containing all necessary data to reconstruct the model.
    """

    model_dict = model.__dict__

    metadata = {
        "init_params": model.get_params(),
        "other_params": {af: model_dict.get(af, None) for af in all_features},
        "version_sklearn_in": version_in,
    }

    return json_convert(metadata)


def _deserialize_knn(model_class, data: dict, ravel_y: bool = False):
    """
    Reconstruct a k-nearest neighbors model from a serialized dictionary.

    Parameters
    ----------
    model_class : type
        KNeighborsRegressor or KNeighborsClassifier.
    data : dict
        Dictionary produced by _serialize_knn.
    ravel_y : bool
        If True, flatten the training targets before fitting.

    Returns
    -------
    KNeighborsRegressor or KNeighborsClassifier
        A reconstructed scikit-learn k-nearest neighbors instance.
    """

    init_params = data["init_params"]
    other_params = data["other_params"]

    X = np.asarray(other_params["_fit_X"])
    y = other_params["_y"]

    if hasattr(y, "values"):
        y = y.values

    y = np.asarray(y)

    if ravel_y:
        y = y.ravel()

    new_model = model_class(**init_params)
    new_model.fit(X, y)

    if "feature_names_in_" in other_params and other_params["feature_names_in_"] is not None:
        new_model.feature_names_in_ = np.array(other_params["feature_names_in_"])

    return new_model
