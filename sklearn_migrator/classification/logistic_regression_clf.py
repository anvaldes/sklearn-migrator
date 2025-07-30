import numpy as np
from sklearn.linear_model import LogisticRegression

all_features = [
    'warm_start',
    'penalty',
    'dual',
    'class_weight',
    'n_jobs',
    'max_iter',
    'fit_intercept',
    'intercept_scaling',
    'multi_class',
    'solver',
    'verbose',
    'C',
    'l1_ratio',
    'tol',
    'n_features_in_', 
    'feature_names_in_'
]

def serialize_logistic_regression_clf(model, version_in):

    metadata = {
        'classes_': model.classes_.tolist(),
        'coef_': model.coef_.tolist(),
        'intercept_': model.intercept_.tolist(),
        'n_iter_': model.n_iter_.tolist(),
        'params': model.get_params(),
        'version_sklearn_in': version_in
    }

    model_dict = model.__dict__
    model_dict_keys = list(model_dict.keys())

    default_values = {
        'n_features_in_': len(model.coef_.tolist()[0]),
        'feature_names_in_': None
    }

    kdv = list(default_values.keys())

    other_params = {}

    for af in all_features:
        if (af in model_dict_keys) == False:
            other_params[af] = default_values[af]
        else:
            other_params[af] = model_dict[af]

    if other_params['multi_class'] == 'deprecated':
        del other_params['multi_class']

    metadata['other_params'] = other_params

    return metadata


def deserialize_logistic_regression_clf(data, version_out):

    model = LogisticRegression(data['params'])
    
    model.classes_ = np.array(data['classes_'])
    model.coef_ = np.array(data['coef_'])
    model.intercept_ = np.array(data['intercept_'])
    model.n_iter_ = np.array(data['n_iter_'])

    for af in all_features:
        try:
            model.__dict__[af] = data['other_params'][af]
        except:
            pass

    return model