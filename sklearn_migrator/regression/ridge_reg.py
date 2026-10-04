from sklearn.linear_model import Ridge
from ._linear import all_features  # noqa: F401
from ._linear import _serialize_penalized_linear, _deserialize_penalized_linear


def serialize_ridge_reg(model: Ridge, version_in: str) -> dict:
    """
    Serialize a fitted Ridge regression model into a JSON-compatible dictionary.

    Parameters
    ----------
    model : Ridge
        A fitted scikit-learn Ridge instance.
    version_in : str
        The sklearn version used to train the model (e.g. '1.2.0').

    Returns
    -------
    dict
        A dictionary containing all necessary data to reconstruct the model.
    """

    return _serialize_penalized_linear(model, version_in)


def deserialize_ridge_reg(data: dict, version_out: str) -> Ridge:
    """
    Reconstruct a Ridge regression model from a serialized dictionary.

    Parameters
    ----------
    data : dict
        Dictionary produced by serialize_ridge_reg.
    version_out : str
        The sklearn version of the target environment (e.g. '1.7.0').

    Returns
    -------
    Ridge
        A reconstructed scikit-learn Ridge instance.
    """

    return _deserialize_penalized_linear(Ridge, data)
