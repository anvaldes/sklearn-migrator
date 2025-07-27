import numpy as np
from sklearn.linear_model import LinearRegression

all_features = [ 
    'fit_intercept', 
    'n_jobs', 
    'copy_X',
    'normalize',
    'n_features_in_',
    'positive',
    'feature_names_in_',
    'tol'
]

def serialize_linear_regression_reg(model, version_in):

    metadata = {
        'coef_': model.coef_.tolist(),
        'singular_': model.singular_.tolist(),
        'intercept_': model.intercept_.tolist(),
        'rank_': model.rank_,
        'version_sklearn_in': version_in
    }

    model_dict = model.__dict__
    model_dict_keys = list(model_dict.keys())

    default_values = {
        'normalize': False,
        'n_features_in_': len(model.coef_) if model.coef_.ndim == 1 else len(model.coef_[0]),
        'positive': False,
        'feature_names_in_': None,
        'tol': 1e-6
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


def deserialize_linear_regression_reg(data, version_out):

    model = LinearRegression()

    model.coef_ = np.array(data['coef_'])
    model.singular_ = np.array(data['singular_'])
    model.intercept_ = np.array(data['intercept_'])
    model.rank_ = data['rank_']

    for af in all_features:
        try:
            model.__dict__[af] = data['other_params'][af]
        except:
            pass

    return model