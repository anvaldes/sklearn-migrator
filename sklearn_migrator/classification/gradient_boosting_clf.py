import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from ..regression.decision_tree_reg import serialize_decision_tree_reg
from ..regression.decision_tree_reg import deserialize_decision_tree_reg
from sklearn.dummy import DummyClassifier

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
    'classes_',
    'min_impurity_split',
    'ccp_alpha',
    'feature_names_in_',
    'n_trees_per_iteration_',
    'presort'
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
            'deviance': lambda: _gb_losses.BinomialDeviance,
            'log_loss': lambda: _gb_losses.BinomialDeviance,
            'exponential': lambda: _gb_losses.ExponentialLoss,
            'multinomial': lambda: _gb_losses.MultinomialDeviance
        }
        return mapping[loss_str]()
else:

    from sklearn._loss.loss import HalfBinomialLoss, ExponentialLoss, HalfMultinomialLoss

    def get_loss_object(loss_str):
        mapping = {
            'deviance': lambda: HalfBinomialLoss,
            'log_loss': lambda: HalfBinomialLoss,
            'exponential': lambda : ExponentialLoss,
            'multinomial': lambda: HalfMultinomialLoss
        }
        return mapping[loss_str]()


def serialize_gradient_boosting_clf(model, version_in):

    metadata = {}

    estimators = model.estimators_

    estimators_ser = [serialize_decision_tree_reg(e[0], version_in) for e in estimators]
    params = model.get_params()

    metadata['estimators'] = estimators_ser
    metadata['params'] = params

    dummy_classifier = {
        'strategy': model.init_.strategy,
        'n_outputs_': model.init_.n_outputs_,
        'output_2d_': getattr(model.init_, 'output_2d_', False),
        'n_classes_': int(model.init_.n_classes_),
        'classes_': [x.item() if hasattr(x, "item") else x for x in model.init_.classes_],
        'class_prior_': [x.item() if hasattr(x, "item") else x for x in model.init_.class_prior_]
    }

    metadata['dummy_clf'] = dummy_classifier

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


def deserialize_gradient_boosting_clf(data, version_out):

    pre_model = GradientBoostingClassifier()
    pre_get_params = list(pre_model.get_params().keys())

    get_params = {}

    for param in pre_get_params:
        if param in list(data['params'].keys()):
            get_params[param] = data['params'][param]

    new_model = GradientBoostingClassifier(**get_params)

    estimators = [[deserialize_decision_tree_reg(e, version_out)] for e in data['estimators']]

    new_model.estimators_ = np.array(estimators)

    init_model = DummyClassifier(strategy = data['dummy_clf']['strategy'])
    init_model.n_outputs_ = data['dummy_clf']['n_outputs_']
    init_model.output_2d_ = data['dummy_clf']['output_2d_']
    init_model.n_classes_ = data['dummy_clf']['n_classes_']
    init_model.classes_ = data['dummy_clf']['classes_']
    init_model.class_prior_ = data['dummy_clf']['class_prior_']
    init_model._strategy = data['dummy_clf']['strategy']

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
    
    if (version_tuple(version_out) >= version_tuple('0.21.3')) and (version_tuple(version_out) < version_tuple('1.1.0')):
        new_model.loss_ = get_loss_object(data['loss'])(data['dummy_clf']['n_classes_'])
    else:
        new_model._loss = get_loss_object(data['loss'])(data['dummy_clf']['n_classes_'])
    
    new_model.train_score_ = data['train_score_']

    for af in all_features:
        try:
            new_model.__dict__[af] = data['other_params'][af]
        except:
            pass

    return new_model