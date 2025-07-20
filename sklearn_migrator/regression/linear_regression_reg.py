import numpy as np
from sklearn.linear_model import LinearRegression

def serialize_linear_regression_reg(model, version_in):

    init_params = model.get_params()
    keys_init = init_params.keys()

    if ('tol' in keys_init) == False:
        init_params['tol'] = 1e-6

    if ('normalize' in keys_init) == False:
        init_params['normalize'] = False

    if ('positive' in keys_init) == False:
        init_params['positive'] = False

    metadata = {
        'meta': 'linear-regression',
        'coef_': model.coef_.tolist(),
        'intercept_': model.intercept_.tolist(),
        'params': init_params,
        'version_sklearn_in': version_in
    }

    return metadata


def deserialize_linear_regression_reg(data, version_out):

    if (version_out >= '0.21.3') and (version_out <= '0.23.2'):
        valid_keys = ['copy_X', 'fit_intercept', 'n_jobs', 'normalize']
    elif (version_out >= '0.24') and (version_out <= '0.24.2'):
        valid_keys = ['copy_X', 'fit_intercept', 'n_jobs', 'normalize', 'positive']
    elif (version_out >= '1.0') and (version_out <= '1.6.1'):
        valid_keys = ['copy_X', 'fit_intercept', 'n_jobs', 'positive']
    elif (version_out >= '1.7'):
        valid_keys = ['copy_X', 'fit_intercept', 'n_jobs', 'positive', 'tol']
    else:
        raise ValueError(f"Versión de scikit-learn no soportada: {version_out}")

    params = {k: v for k, v in data['params'].items() if k in valid_keys}

    model = LinearRegression(**params)

    model.coef_ = np.array(data['coef_'])
    model.intercept_ = np.array(data['intercept_'])

    return model