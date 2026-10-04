from sklearn.linear_model import Lasso
from ._linear import all_features  # noqa: F401
from ._linear import _serialize_penalized_linear, _deserialize_penalized_linear


def serialize_lasso_reg(model: Lasso, version_in: str) -> dict:
    """
    Serialize a fitted Lasso regression model into a JSON-compatible dictionary.

    Parameters
    ----------
    model : Lasso
        A fitted scikit-learn Lasso instance.
    version_in : str
        The sklearn version used to train the model (e.g. '1.2.0').

    Returns
    -------
    dict
        A dictionary containing all necessary data to reconstruct the model.
    """

    return _serialize_penalized_linear(model, version_in)


def deserialize_lasso_reg(data: dict, version_out: str) -> Lasso:
    """
    Reconstruct a Lasso regression model from a serialized dictionary.

    Parameters
    ----------
    data : dict
        Dictionary produced by serialize_lasso_reg.
    version_out : str
        The sklearn version of the target environment (e.g. '1.7.0').

    Returns
    -------
    Lasso
        A reconstructed scikit-learn Lasso instance.
    """

    return _deserialize_penalized_linear(Lasso, data)
