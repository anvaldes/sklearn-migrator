import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from .decision_tree_reg import serialize_decision_tree_reg
from .decision_tree_reg import deserialize_decision_tree_reg
from sklearn.dummy import DummyRegressor

import sklearn

version_sklearn = sklearn.__version__


all_features = [
    'criterion',
    'init',
    'alpha',
    'min_samples_split',
    'min_impurity_decrease',
    'n_estimators',
    'warm_start',
    'max_depth',
    'max_leaf_nodes',
    'validation_fraction',
    'verbose',
    'max_features',
    'tol',
    'min_weight_fraction_leaf',
    'min_samples_leaf',
    'subsample',
    'learning_rate',
    'n_iter_no_change',
    'max_features_',
    'n_estimators_',
    'n_classes_',
    'presort',
    'min_impurity_split',
    'feature_names_in_',
    'ccp_alpha',
    'n_trees_per_iteration_'
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


if version_tuple(sklearn.__version__) < version_tuple('1.4.0'):

    from sklearn.ensemble import _gb_losses

    def get_loss_object(loss_str):
        mapping = {
            'ls': lambda: _gb_losses.LeastSquaresError,
            'squared_error': lambda: _gb_losses.LeastSquaresError,
            'lad': lambda: _gb_losses.LeastAbsoluteError,
            'huber': lambda: _gb_losses.HuberLossFunction,
            'quantile': lambda: _gb_losses.QuantileLossFunction
        }
        return mapping[loss_str]()
else:

    from sklearn._loss.loss import HalfSquaredError, AbsoluteError, HuberLoss, PinballLoss

    def get_loss_object(loss_str):
        mapping = {
            'ls': lambda: HalfSquaredError,
            'squared_error': lambda : HalfSquaredError,
            'lad': lambda: AbsoluteError,
            'huber': lambda: HuberLoss,
            'quantile': lambda: PinballLoss
        }
        return mapping[loss_str]()


def serialize_gradient_boosting_reg(model, version_in):

    metadata = {}

    estimators = model.estimators_

    estimators_ser = [serialize_decision_tree_reg(e[0], version_in) for e in estimators]
    params = model.get_params()

    metadata['estimators'] = estimators_ser
    metadata['params'] = params

    dummy_regressor = {
        'strategy': model.init_.strategy,
        'n_outputs_': model.init_.n_outputs_,
        'constant_': float(model.init_.constant_[0][0]),
        'output_2d_': getattr(model.init_, 'output_2d_', False)
    }

    metadata['dummy_reg'] = dummy_regressor

    metadata['loss'] = model.loss

    try:
        metadata['n_features_in'] = model.n_features_in_
    except:
        metadata['n_features_in'] = None
    
    try:
        metadata['n_features'] = model.n_features_
    except:
        metadata['n_features'] = None

    metadata['train_score_'] = list(model.train_score_)

    model_dict = model.__dict__
    model_dict_keys = list(model_dict.keys())

    default_values = {
        'min_impurity_split': None,
        'ccp_alpha': 0.0,
        'feature_names_in_': None,
        'n_classes_': 1,
        'presort': 'auto',
        'n_trees_per_iteration_': 1
    }

    kdv = list(default_values.keys())

    other_params = {}

    for af in all_features:
        if (af in model_dict_keys) == False:
            other_params[af] = default_values[af]
        else:
            other_params[af] = model_dict[af]

    metadata['other_params'] = other_params
    metadata['version_sklearn_in'] = version_in

    return metadata


def deserialize_gradient_boosting_reg(data, version_out):

    pre_model = GradientBoostingRegressor()
    pre_get_params = list(pre_model.get_params().keys())

    get_params = {}

    for param in pre_get_params:
        if param in list(data['params'].keys()):
            get_params[param] = data['params'][param]

    new_model = GradientBoostingRegressor(**get_params)

    estimators = [[deserialize_decision_tree_reg(e, version_out)] for e in data['estimators']]

    new_model.estimators_ = np.array(estimators)

    init_model = DummyRegressor(strategy = data['dummy_reg']['strategy'])
    init_model.n_outputs_ = data['dummy_reg']['n_outputs_']
    init_model.constant_ = np.array([data['dummy_reg']['constant_']])
    init_model.output_2d_ = data['dummy_reg']['output_2d_']

    new_model.init_ = init_model

    n_features = (data['n_features'] or data['n_features_in'])

    try:
        new_model.n_features_ = n_features
    except:
        pass

    try:
        new_model.n_features_in_ = n_features
    except:
        pass
    
    
    if (version_tuple(version_out) >= version_tuple('0.21.3')) and (version_tuple(version_out) <= version_tuple('0.23.2')):
        new_model.loss_ = get_loss_object(data['loss'])(1)
    elif (version_tuple(version_out) > version_tuple('0.23.2')) and (version_tuple(version_out) < version_tuple('1.1.0')):
        new_model.loss_ = get_loss_object(data['loss'])()
    else:
        new_model._loss = get_loss_object(data['loss'])()
    
    new_model.train_score_ = data['train_score_']

    for af in all_features:
        try:
            new_model.__dict__[af] = data['other_params'][af]
        except:
            pass

    return new_model