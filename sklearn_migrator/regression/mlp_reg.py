from sklearn.neural_network import MLPRegressor
from .._mlp import all_features  # noqa: F401
from .._mlp import _serialize_mlp, _deserialize_mlp


def serialize_mlp_reg(model: MLPRegressor, version_in: str) -> dict:
    """
    Serialize a fitted MLPRegressor into a JSON-compatible dictionary.

    Parameters
    ----------
    model : MLPRegressor
        A fitted scikit-learn MLPRegressor instance.
    version_in : str
        The sklearn version used to train the model (e.g. '1.2.0').

    Returns
    -------
    dict
        A dictionary containing all necessary data to reconstruct the model.
    """

    return _serialize_mlp(model, version_in, 'mlp-regression')


def deserialize_mlp_reg(data: dict, version_out: str) -> MLPRegressor:
    """
    Reconstruct a MLPRegressor from a serialized dictionary.

    Parameters
    ----------
    data : dict
        Dictionary produced by serialize_mlp_reg.
    version_out : str
        The sklearn version of the target environment (e.g. '1.7.0').

    Returns
    -------
    MLPRegressor
        A reconstructed scikit-learn MLPRegressor instance.
    """

    return _deserialize_mlp(MLPRegressor, data)
