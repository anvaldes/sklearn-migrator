import numpy as np
from sklearn.linear_model import Lasso

all_features = [ 
    'fit_intercept', 
    'copy_X',
    'n_features_in_',
    'feature_names_in_',
    'tol',
    'n_iter_'
]

def serialize_lasso_reg(model, version_in):

    metadata = {
        'alpha': model.alpha,
        'coef_': model.coef_.tolist(),
        'intercept_': model.intercept_.tolist(),
        'version_sklearn_in': version_in
    }

    model_dict = model.__dict__
    model_dict_keys = list(model_dict.keys())

    default_values = {
        'n_features_in_': len(model.coef_) if model.coef_.ndim == 1 else len(model.coef_[0]),
        'feature_names_in_': None,
        'tol': 1e-6,
        'n_iter_': 1
        }
    
    kdv = list(default_values.keys())

    other_params = {}

    for af in all_features:
        if (af in model_dict_keys) == False:
            other_params[af] = default_values[af]
        else:
            other_params[af] = model_dict[af]

    metadata['other_params'] = other_params

    return metadata


def deserialize_lasso_reg(data, version_out):

    model = Lasso()

    model.alpha = data['alpha']
    model.coef_ = np.array(data['coef_'])
    model.intercept_ = np.array(data['intercept_'])

    for af in all_features:
        try:
            model.__dict__[af] = data['other_params'][af]
        except:
            pass

    return model