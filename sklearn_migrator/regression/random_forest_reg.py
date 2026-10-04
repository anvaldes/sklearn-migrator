from sklearn.ensemble import RandomForestRegressor
from .decision_tree_reg import serialize_decision_tree_reg
from .decision_tree_reg import deserialize_decision_tree_reg
from .._forest import _serialize_forest, _deserialize_forest
from ..utils import set_attr_safely

all_features = [
    'n_estimators',
    'n_outputs_',
    'oob_score',
    'min_weight_fraction_leaf',
    'verbose',
    'warm_start',
    'min_samples_leaf',
    'criterion',
    'min_samples_split',
    'class_weight',
    'min_impurity_decrease',
    'max_features',
    'max_leaf_nodes',
    'n_jobs',
    'max_depth',
    'bootstrap',
    'min_impurity_split',
    'max_samples',
    'ccp_alpha',
    'feature_names_in_',
    'monotonic_cst',
]


def serialize_random_forest_reg(model: RandomForestRegressor, version_in: str) -> dict:
    """
    Serialize a fitted RandomForestRegressor into a JSON-compatible dictionary.

    Parameters
    ----------
    model : RandomForestRegressor
        A fitted scikit-learn RandomForestRegressor instance.
    version_in : str
        The sklearn version used to train the model (e.g. '1.2.0').

    Returns
    -------
    dict
        A dictionary containing all necessary data to reconstruct the model.
    """

    return _serialize_forest(model, version_in, serialize_decision_tree_reg, all_features)


def deserialize_random_forest_reg(data: dict, version_out: str) -> RandomForestRegressor:
    """
    Reconstruct a RandomForestRegressor from a serialized dictionary.

    Parameters
    ----------
    data : dict
        Dictionary produced by serialize_random_forest_reg.
    version_out : str
        The sklearn version of the target environment (e.g. '1.7.0').

    Returns
    -------
    RandomForestRegressor
        A reconstructed scikit-learn RandomForestRegressor instance.
    """

    new_model, n_features = _deserialize_forest(
        RandomForestRegressor, data, version_out, deserialize_decision_tree_reg, all_features
    )

    set_attr_safely(new_model, 'n_features_', n_features)
    set_attr_safely(new_model, 'n_features_in_', n_features)

    return new_model
