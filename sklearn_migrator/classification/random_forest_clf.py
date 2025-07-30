import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .decision_tree_clf import serialize_decision_tree_clf
from .decision_tree_clf import deserialize_decision_tree_clf

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

def serialize_random_forest_clf(model, version_in):

    estimators = model.estimators_
    estimators_ser = [serialize_decision_tree_clf(e, version_in) for e in estimators]
    params = model.get_params()

    metadata = {}

    metadata['estimators'] = estimators_ser
    metadata['params'] = params
    metadata['estimator_params'] = model.estimator_params

    model_dict = model.__dict__
    model_dict_keys = list(model_dict.keys())

    default_values = {
        'min_impurity_split': None,
        'max_samples': None,
        'ccp_alpha': 0.0,
        'feature_names_in_': None,
        'monotonic_cst': None
    }

    kdv = list(default_values.keys())

    other_params = {}

    for af in all_features:
        if (af in model_dict_keys) == False:
            other_params[af] = default_values[af]
        else:
            other_params[af] = model_dict[af]

    try:
        other_params['n_features'] = model.n_features_
    except:
        other_params['n_features'] = None

    try:
        other_params['n_features_in'] = model.n_features_in_
    except:
        other_params['n_features_in'] = None

    metadata['other_params'] = other_params
    metadata['version_sklearn_in'] = version_in

    return metadata


def deserialize_random_forest_clf(data, version_out):

    pre_model = RandomForestClassifier()
    pre_get_params = list(pre_model.get_params().keys())

    get_params = {}

    for param in pre_get_params:
        if param in list(data['params'].keys()):
            get_params[param] = data['params'][param]

    new_model = RandomForestClassifier(**get_params)

    estimators = [deserialize_decision_tree_clf(e, version_out) for e in data['estimators']]

    new_model.estimators_ = estimators
    new_model.estimator_params = data['estimator_params']

    for af in all_features:
        try:
            new_model.__dict__[af] = data['other_params'][af]
        except:
            pass

    n_features = (data['other_params']['n_features'] or data['other_params']['n_features_in'])

    try:
        new_model.n_features_ = n_features
    except:
        pass

    try:
        new_model.n_features_in_ = n_features
    except:
        pass

    try:
        new_model.base_estimator = RandomForestClassifier()
    except:
        pass

    try:
        new_model.base_estimator_ = RandomForestClassifier()
    except:
        pass

    try:
        new_model.estimator = RandomForestClassifier()
    except:
        pass

    try:
        new_model._estimator = RandomForestClassifier()
    except:
        pass

    try:
        new_model.estimator_ = RandomForestClassifier()
    except:
        pass

    return new_model