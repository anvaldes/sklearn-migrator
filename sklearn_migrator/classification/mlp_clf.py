from sklearn.neural_network import MLPClassifier
from .._mlp import all_features  # noqa: F401
from .._mlp import _serialize_mlp, _deserialize_mlp


def serialize_mlp_clf(model: MLPClassifier, version_in: str) -> dict:
    """
    Serialize a fitted MLPClassifier into a JSON-compatible dictionary.

    Parameters
    ----------
    model : MLPClassifier
        A fitted scikit-learn MLPClassifier instance.
    version_in : str
        The sklearn version used to train the model (e.g. '1.2.0').

    Returns
    -------
    dict
        A dictionary containing all necessary data to reconstruct the model.
    """

    return _serialize_mlp(model, version_in, 'mlp-classifier')


def deserialize_mlp_clf(data: dict, version_out: str) -> MLPClassifier:
    """
    Reconstruct a MLPClassifier from a serialized dictionary.

    Parameters
    ----------
    data : dict
        Dictionary produced by serialize_mlp_clf.
    version_out : str
        The sklearn version of the target environment (e.g. '1.7.0').

    Returns
    -------
    MLPClassifier
        A reconstructed scikit-learn MLPClassifier instance.
    """

    return _deserialize_mlp(MLPClassifier, data)
