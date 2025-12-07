import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from ..regression.decision_tree_reg import serialize_decision_tree_reg
from ..regression.decision_tree_reg import deserialize_decision_tree_reg

import sklearn

version_sklearn = sklearn.__version__


all_features = [
    'n_features_in_',
    'feature_names_in_',
    'n_features_'
]


def version_tuple(version):

    version_split = version.split('.')

    if len(version_split) == 1:
        new_version = (int(version_split[0]), 0, 0)
    elif len(version_split) == 2:
        new_version = (int(version_split[0]), int(version_split[1]), 0)
    elif len(version_split) == 3:
        new_version = (int(version_split[0]), int(version_split[1]), int(version_split[2]))
    else:
        new_version = 'Formato no valido'

    return new_version


def serialize_adaboost_reg(model, version_in):

    metadata = {}

    estimators = model.estimators_
    estimators_ser = [serialize_decision_tree_reg(e, version_in) for e in estimators]
    params = model.get_params()

    metadata['estimators'] = estimators_ser
    metadata['params'] = params

    metadata['estimator_weights_'] = list(model.estimator_weights_)
    metadata['estimator_errors_'] = list(model.estimator_errors_)

    try:
        metadata['n_features_in'] = int(model.n_features_in_)
    except:
        metadata['n_features_in'] = None

    try:
        metadata['n_features'] = int(getattr(model, 'n_features_', metadata['n_features_in']))
    except:
        metadata['n_features'] = metadata['n_features_in']

    model_dict = model.__dict__
    model_dict_keys = list(model_dict.keys())

    default_values = {
        'n_features_in_': None,
        'feature_names_in_': None,
        'n_features_': None
    }

    other_params = {}

    for af in all_features:
        if (af in model_dict_keys) == False:
            other_params[af] = default_values[af]
        else:
            other_params[af] = model_dict[af]

    metadata['other_params'] = other_params
    metadata['version_sklearn_in'] = version_in

    return metadata


def deserialize_adaboost_reg(data, version_out):

    pre_model = AdaBoostRegressor()
    pre_get_params = list(pre_model.get_params().keys())

    get_params = {}

    for param in pre_get_params:
        if param in list(data['params'].keys()):
            get_params[param] = data['params'][param]

    if ('estimator' in get_params) and ('base_estimator' in pre_get_params):
        get_params['base_estimator'] = get_params.pop('estimator')
    elif ('base_estimator' in get_params) and ('estimator' in pre_get_params):
        get_params['estimator'] = get_params.pop('base_estimator')

    new_model = AdaBoostRegressor(**get_params)

    estimators = [deserialize_decision_tree_reg(e, version_out) for e in data['estimators']]
    new_model.estimators_ = np.array(estimators, dtype=object)

    new_model.estimator_weights_ = np.array(data['estimator_weights_'])
    new_model.estimator_errors_ = np.array(data['estimator_errors_'])

    n_features = data.get('n_features') or data.get('n_features_in')

    if n_features is None and len(estimators) > 0:
        est0 = estimators[0]
        n_features = getattr(
            est0,
            "n_features_in_",
            getattr(est0, "n_features_", None)
        )
        if isinstance(n_features, np.integer):
            n_features = int(n_features)

    if n_features is not None:
        try:
            new_model.n_features_ = n_features
        except Exception:
            pass

        try:
            new_model.n_features_in_ = n_features
        except Exception:
            pass

    for af in all_features:
        try:
            val = data['other_params'][af]
        except KeyError:
            continue

        if af in ['n_features_in_', 'n_features_'] and (val is None or n_features is not None):
            continue

        new_model.__dict__[af] = val

    return new_model
