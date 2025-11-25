import numpy as np
from sklearn.neural_network import MLPClassifier

all_features = [
    'batch_size',
    'best_validation_score_',
    'feature_names_in_',
    'max_fun',
    'validation_scores_',
    'best_validation_score_',
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

def serialize_mlp_clf(model, version_in):

    metadata = {}

    params = model.get_params()

    del_var = ['max_fun', 'loss']

    for d_v in del_var:
        try:
            del params[d_v]
        except:
            pass

    serialized_mlp = {
            'meta': 'mlp-classifier',
            'coefs_': [c.tolist() for c in model.coefs_],
            'loss_': float(model.loss_),
            'intercepts_': [b.tolist() for b in model.intercepts_],
            'n_iter_': int(model.n_iter_),
            'n_layers_': int(model.n_layers_),
            'n_outputs_': int(model.n_outputs_),
            'out_activation_': model.out_activation_,
            'params': params
        }

    metadata['serialized_mlp'] = serialized_mlp

    model_dict = model.__dict__
    model_dict_keys = list(model_dict.keys())

    default_values = {
        'best_validation_score_': None,
        'feature_names_in_': None,
        'max_fun': 15000,
        'validation_scores_': None,
        'best_validation_score_': None,
        'n_features_in_': len(model.coefs_[0])
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


def deserialize_mlp_clf(data, version_out):
  
    version_in = data['version_sklearn_in']
    serialized_mlp = data['serialized_mlp']
  
    new_model = MLPClassifier(**serialized_mlp['params'])
    
    new_model.coefs_ = [np.array(c) for c in serialized_mlp['coefs_']]
    new_model.intercepts_ = [np.array(b) for b in serialized_mlp['intercepts_']]
    
    new_model.loss_ = serialized_mlp['loss_']
    new_model.n_iter_ = serialized_mlp['n_iter_']
    new_model.n_layers_ = serialized_mlp['n_layers_']
    new_model.n_outputs_ = serialized_mlp['n_outputs_']
    new_model.out_activation_ = serialized_mlp['out_activation_']

    for af in all_features:
        try:
            new_model.__dict__[af] = data['other_params'][af]
        except:
           pass
    
    return new_model