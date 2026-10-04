from .utils import json_convert, get_attr_or_none
from .utils import collect_other_params, restore_other_params, filter_init_params


def _serialize_forest(model, version_in: str, serialize_tree, all_features: list) -> dict:
    """
    Serialize a fitted random forest into a JSON-compatible dictionary.

    Parameters
    ----------
    model : RandomForestRegressor or RandomForestClassifier
        A fitted scikit-learn random forest instance.
    version_in : str
        The sklearn version used to train the model.
    serialize_tree : callable
        Function used to serialize each tree of the forest.
    all_features : list
        Names of the fields to read from the model.

    Returns
    -------
    dict
        A dictionary containing all necessary data to reconstruct the model.
    """

    metadata = {}

    metadata['estimators'] = [serialize_tree(e, version_in) for e in model.estimators_]
    metadata['params'] = model.get_params()
    metadata['estimator_params'] = model.estimator_params

    default_values = {
        'min_impurity_split': None,
        'max_samples': None,
        'ccp_alpha': 0.0,
        'feature_names_in_': None,
        'monotonic_cst': None
    }

    other_params = collect_other_params(model, all_features, default_values)
    other_params['n_features'] = get_attr_or_none(model, 'n_features_')
    other_params['n_features_in'] = get_attr_or_none(model, 'n_features_in_')

    metadata['other_params'] = other_params
    metadata['version_sklearn_in'] = version_in

    return json_convert(metadata)


def _deserialize_forest(model_class, data: dict, version_out: str, deserialize_tree, all_features: list):
    """
    Reconstruct a random forest from a serialized dictionary.

    Parameters
    ----------
    model_class : type
        RandomForestRegressor or RandomForestClassifier.
    data : dict
        Dictionary produced by _serialize_forest.
    version_out : str
        The sklearn version of the target environment.
    deserialize_tree : callable
        Function used to reconstruct each tree of the forest.
    all_features : list
        Names of the fields to set on the model.

    Returns
    -------
    tuple
        A tuple of (model, n_features) with the reconstructed forest and its
        number of features, which the caller still has to set on the model.
    """

    new_model = model_class(**filter_init_params(model_class, data['params']))

    new_model.estimators_ = [deserialize_tree(e, version_out) for e in data['estimators']]
    new_model.estimator_params = data['estimator_params']

    restore_other_params(new_model, all_features, data)

    n_features = (data['other_params']['n_features'] or data['other_params']['n_features_in'])

    return new_model, n_features
