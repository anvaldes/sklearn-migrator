from sklearn.ensemble import RandomForestClassifier
from .decision_tree_clf import serialize_decision_tree_clf
from .decision_tree_clf import deserialize_decision_tree_clf
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
    'classes_',
    'n_classes_',
    'min_impurity_split',
    'max_samples',
    'ccp_alpha',
    'feature_names_in_',
    'monotonic_cst',
]

# Names under which the different sklearn versions store the base estimator.
_estimator_fields = [
    'base_estimator',
    'base_estimator_',
    'estimator',
    '_estimator',
    'estimator_',
]


def serialize_random_forest_clf(model: RandomForestClassifier, version_in: str) -> dict:
    """
    Serialize a fitted RandomForestClassifier into a JSON-compatible dictionary.

    Parameters
    ----------
    model : RandomForestClassifier
        A fitted scikit-learn RandomForestClassifier instance.
    version_in : str
        The sklearn version used to train the model (e.g. '1.2.0').

    Returns
    -------
    dict
        A dictionary containing all necessary data to reconstruct the model.
    """

    return _serialize_forest(model, version_in, serialize_decision_tree_clf, all_features)


def deserialize_random_forest_clf(data: dict, version_out: str) -> RandomForestClassifier:
    """
    Reconstruct a RandomForestClassifier from a serialized dictionary.

    Parameters
    ----------
    data : dict
        Dictionary produced by serialize_random_forest_clf.
    version_out : str
        The sklearn version of the target environment (e.g. '1.7.0').

    Returns
    -------
    RandomForestClassifier
        A reconstructed scikit-learn RandomForestClassifier instance.
    """

    new_model, n_features = _deserialize_forest(
        RandomForestClassifier, data, version_out, deserialize_decision_tree_clf, all_features
    )

    set_attr_safely(new_model, 'n_features_', n_features, warn=True)
    set_attr_safely(new_model, 'n_features_in_', n_features, warn=True)

    for field in _estimator_fields:
        set_attr_safely(new_model, field, RandomForestClassifier(), warn=True)

    return new_model
