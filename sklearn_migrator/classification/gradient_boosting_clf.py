import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from ..regression.decision_tree_reg import serialize_decision_tree_reg
from ..regression.decision_tree_reg import deserialize_decision_tree_reg
from sklearn.dummy import DummyClassifier
from ..utils import json_convert, version_tuple, get_attr_or_none, set_attr_safely
from ..utils import collect_other_params, restore_other_params, filter_init_params

import sklearn


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


if version_tuple(sklearn.__version__) < version_tuple('1.4.0'):

    from sklearn.ensemble import _gb_losses

    def get_loss_object(loss_str: str) -> type:
        """
        Return the appropriate loss class for the given loss string.

        Parameters
        ----------
        loss_str : str
            Loss function name (e.g. 'log_loss', 'exponential', 'multinomial').

        Returns
        -------
        type
            The corresponding loss class for the current sklearn version.
        """

        mapping = {
            'deviance': lambda: _gb_losses.BinomialDeviance,
            'log_loss': lambda: _gb_losses.BinomialDeviance,
            'exponential': lambda: _gb_losses.ExponentialLoss,
            'multinomial': lambda: _gb_losses.MultinomialDeviance
        }
        return mapping[loss_str]()
else:

    from sklearn._loss.loss import HalfBinomialLoss, ExponentialLoss, HalfMultinomialLoss

    def get_loss_object(loss_str: str) -> type:
        """
        Return the appropriate loss class for the given loss string.

        Parameters
        ----------
        loss_str : str
            Loss function name (e.g. 'log_loss', 'exponential', 'multinomial').

        Returns
        -------
        type
            The corresponding loss class for the current sklearn version.
        """

        mapping = {
            'deviance': lambda: HalfBinomialLoss,
            'log_loss': lambda: HalfBinomialLoss,
            'exponential': lambda : ExponentialLoss,
            'multinomial': lambda: HalfMultinomialLoss
        }
        return mapping[loss_str]()


def serialize_gradient_boosting_clf(model: GradientBoostingClassifier, version_in: str) -> dict:
    """
    Serialize a fitted GradientBoostingClassifier into a JSON-compatible dictionary.

    Parameters
    ----------
    model : GradientBoostingClassifier
        A fitted scikit-learn GradientBoostingClassifier instance.
    version_in : str
        The sklearn version used to train the model (e.g. '1.2.0').

    Returns
    -------
    dict
        A dictionary containing all necessary data to reconstruct the model.
    """

    metadata = {}

    metadata['estimators'] = [serialize_decision_tree_reg(e[0], version_in) for e in model.estimators_]
    metadata['params'] = model.get_params()

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

    metadata['n_features_in'] = get_attr_or_none(model, 'n_features_in_')
    metadata['n_features'] = get_attr_or_none(model, 'n_features_')

    metadata['train_score_'] = list(model.train_score_)

    default_values = {
        'min_impurity_split': None,
        'ccp_alpha': 0.0,
        'feature_names_in_': None,
        'presort': 'auto',
        'n_trees_per_iteration_': 1
    }

    metadata['other_params'] = collect_other_params(model, all_features, default_values)
    metadata['version_sklearn_in'] = version_in

    return json_convert(metadata)

def deserialize_gradient_boosting_clf(data: dict, version_out: str) -> GradientBoostingClassifier:
    """
    Reconstruct a GradientBoostingClassifier from a serialized dictionary.

    Parameters
    ----------
    data : dict
        Dictionary produced by serialize_gradient_boosting_clf.
    version_out : str
        The sklearn version of the target environment (e.g. '1.7.0').

    Returns
    -------
    GradientBoostingClassifier
        A reconstructed scikit-learn GradientBoostingClassifier instance.
    """

    new_model = GradientBoostingClassifier(**filter_init_params(GradientBoostingClassifier, data['params']))

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

    set_attr_safely(new_model, 'n_features_', n_features, warn=True)
    set_attr_safely(new_model, 'n_features_in_', n_features, warn=True)

    if (version_tuple(version_out) >= version_tuple('0.21.3')) and (version_tuple(version_out) < version_tuple('1.1.0')):
        new_model.loss_ = get_loss_object(data['loss'])(data['dummy_clf']['n_classes_'])
    else:
        new_model._loss = get_loss_object(data['loss'])(data['dummy_clf']['n_classes_'])

    new_model.train_score_ = data['train_score_']

    restore_other_params(new_model, all_features, data)

    return new_model