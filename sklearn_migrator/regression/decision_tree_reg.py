import numpy as np
from sklearn.tree import DecisionTreeRegressor
from .._tree import _serialize_tree_state, _load_tree_arrays, _build_tree
from ..utils import json_convert, get_attr_or_none, set_attr_safely
from ..utils import collect_other_params, restore_other_params

all_features = [
    'max_leaf_nodes',
    'max_features_',
    'min_weight_fraction_leaf',
    'splitter',
    'class_weight',
    'min_impurity_decrease',
    'min_samples_split',
    'criterion',
    'min_samples_leaf',
    'max_depth',
    'n_outputs_',
    'max_features',
    'min_impurity_split',
    'n_classes_',
    'classes_',
    'presort',
    'ccp_alpha',
    'feature_names_in_',
    'monotonic_cst'
]


def _get_metadata(model: DecisionTreeRegressor, version_in: str) -> dict:
    """
    Extract feature metadata from a fitted DecisionTreeRegressor.

    Parameters
    ----------
    model : DecisionTreeRegressor
        A fitted scikit-learn DecisionTreeRegressor instance.
    version_in : str
        The sklearn version used to train the model.

    Returns
    -------
    dict
        Dictionary with n_features_in, n_features, n_classes and n_outputs.
    """

    return {
        'n_features_in': get_attr_or_none(model, 'n_features_in_'),
        'n_features': get_attr_or_none(model, 'n_features_'),
        'n_classes': get_attr_or_none(model, 'n_classes_'),
        'n_outputs': model.n_outputs_
    }


def serialize_decision_tree_reg(model: DecisionTreeRegressor, version_in: str) -> dict:
    """
    Serialize a fitted DecisionTreeRegressor into a JSON-compatible dictionary.

    Parameters
    ----------
    model : DecisionTreeRegressor
        A fitted scikit-learn DecisionTreeRegressor instance.
    version_in : str
        The sklearn version used to train the model (e.g. '1.2.0').

    Returns
    -------
    dict
        A dictionary containing all necessary data to reconstruct the model.
    """

    serialized_tree = _serialize_tree_state(model, version_in)

    metadata = _get_metadata(model, version_in)
    metadata['serialized_tree'] = serialized_tree
    metadata['version_sklearn_in'] = version_in

    default_values = {
        'min_impurity_split': None,
        'n_classes_': 1,
        'classes_': None,
        'presort': False,
        'ccp_alpha': 0.0,
        'feature_names_in_': None,
        'monotonic_cst': None
    }

    metadata['other_params'] = collect_other_params(model, all_features, default_values)

    return json_convert(metadata)


def deserialize_decision_tree_reg(data: dict, version_out: str) -> DecisionTreeRegressor:
    """
    Reconstruct a DecisionTreeRegressor from a serialized dictionary.

    Parameters
    ----------
    data : dict
        Dictionary produced by serialize_decision_tree_reg.
    version_out : str
        The sklearn version of the target environment (e.g. '1.7.0').

    Returns
    -------
    DecisionTreeRegressor
        A reconstructed scikit-learn DecisionTreeRegressor instance.
    """

    nodes_array, values_array = _load_tree_arrays(data, version_out)

    n_classes = np.array([1], dtype=np.intp)  # regression
    n_features = (data['n_features'] or data['n_features_in'])

    new_tree = DecisionTreeRegressor(max_depth=data['serialized_tree']['max_depth'], random_state=42)
    new_tree.tree_ = _build_tree(data, nodes_array, values_array, n_classes)
    new_tree.n_outputs_ = data['n_outputs']

    set_attr_safely(new_tree, 'n_features_', n_features)
    set_attr_safely(new_tree, 'n_features_in_', n_features)

    restore_other_params(new_tree, all_features, data)

    return new_tree
