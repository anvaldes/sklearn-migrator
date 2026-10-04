from sklearn.neighbors import KNeighborsClassifier
from .._knn import _serialize_knn, _deserialize_knn

all_features = [
    "_fit_X",
    "_y",
    "classes_",
    "feature_names_in_",
]


def serialize_knn_clf(model: KNeighborsClassifier, version_in: str) -> dict:
    """
    Serialize a fitted KNeighborsClassifier into a JSON-compatible dictionary.

    Parameters
    ----------
    model : KNeighborsClassifier
        A fitted scikit-learn KNeighborsClassifier instance.
    version_in : str
        The sklearn version used to train the model (e.g. '1.2.0').

    Returns
    -------
    dict
        A dictionary containing all necessary data to reconstruct the model.
    """

    return _serialize_knn(model, version_in, all_features)


def deserialize_knn_clf(data: dict, version_out: str) -> KNeighborsClassifier:
    """
    Reconstruct a KNeighborsClassifier from a serialized dictionary.

    Parameters
    ----------
    data : dict
        Dictionary produced by serialize_knn_clf.
    version_out : str
        The sklearn version of the target environment (e.g. '1.7.0').

    Returns
    -------
    KNeighborsClassifier
        A reconstructed scikit-learn KNeighborsClassifier instance.
    """

    return _deserialize_knn(KNeighborsClassifier, data, ravel_y=True)
