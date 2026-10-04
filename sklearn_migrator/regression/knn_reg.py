from sklearn.neighbors import KNeighborsRegressor
from .._knn import _serialize_knn, _deserialize_knn

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

    return _serialize_knn(model, version_in, all_features)


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

    return _deserialize_knn(KNeighborsRegressor, data)
