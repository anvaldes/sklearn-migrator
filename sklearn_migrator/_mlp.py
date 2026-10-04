import numpy as np
from .utils import json_convert, collect_other_params, restore_other_params

all_features = [
    'batch_size',
    'best_validation_score_',
    'feature_names_in_',
    'max_fun',
    'validation_scores_',
    'n_features_in_',
    '_no_improvement_count',
    'activation',
    'alpha',
    'best_loss_',
    'beta_1',
    'beta_2',
    'early_stopping',
    'epsilon',
    'hidden_layer_sizes',
    'learning_rate',
    'learning_rate_init',
    'loss',
    'max_iter',
    'momentum',
    'n_iter_no_change',
    'nesterovs_momentum',
    'power_t',
    'shuffle',
    'solver',
    'tol',
    'validation_fraction',
    'verbose',
    'warm_start'
    ]


def _serialize_mlp(model, version_in: str, meta: str) -> dict:
    """
    Serialize a fitted multi-layer perceptron into a JSON-compatible dictionary.

    Parameters
    ----------
    model : MLPRegressor or MLPClassifier
        A fitted scikit-learn multi-layer perceptron instance.
    version_in : str
        The sklearn version used to train the model.
    meta : str
        Label identifying the kind of model in the serialized dictionary.

    Returns
    -------
    dict
        A dictionary containing all necessary data to reconstruct the model.
    """

    metadata = {}

    params = model.get_params()

    for d_v in ['max_fun', 'loss']:
        params.pop(d_v, None)

    metadata['serialized_mlp'] = {
            'meta': meta,
            'coefs_': [c.tolist() for c in model.coefs_],
            'loss_': float(model.loss_),
            'intercepts_': [b.tolist() for b in model.intercepts_],
            'n_iter_': int(model.n_iter_),
            'n_layers_': int(model.n_layers_),
            'n_outputs_': int(model.n_outputs_),
            'out_activation_': model.out_activation_,
            'params': params
        }

    default_values = {
        'best_validation_score_': None,
        'feature_names_in_': None,
        'max_fun': 15000,
        'validation_scores_': None,
        'n_features_in_': len(model.coefs_[0])
    }

    metadata['other_params'] = collect_other_params(model, all_features, default_values)
    metadata['version_sklearn_in'] = version_in

    return json_convert(metadata)


def _deserialize_mlp(model_class, data: dict):
    """
    Reconstruct a multi-layer perceptron from a serialized dictionary.

    Parameters
    ----------
    model_class : type
        MLPRegressor or MLPClassifier.
    data : dict
        Dictionary produced by _serialize_mlp.

    Returns
    -------
    MLPRegressor or MLPClassifier
        A reconstructed scikit-learn multi-layer perceptron instance.
    """

    serialized_mlp = data['serialized_mlp']

    new_model = model_class(**serialized_mlp['params'])

    new_model.coefs_ = [np.array(c) for c in serialized_mlp['coefs_']]
    new_model.intercepts_ = [np.array(b) for b in serialized_mlp['intercepts_']]

    new_model.loss_ = serialized_mlp['loss_']
    new_model.n_iter_ = serialized_mlp['n_iter_']
    new_model.n_layers_ = serialized_mlp['n_layers_']
    new_model.n_outputs_ = serialized_mlp['n_outputs_']
    new_model.out_activation_ = serialized_mlp['out_activation_']

    restore_other_params(new_model, all_features, data)

    return new_model
