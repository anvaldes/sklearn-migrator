import numpy as np
from sklearn.linear_model import LogisticRegression

def serialize_logistic_regression_clf(model, version_in):
    serialized_model = {
        'meta': 'lr',
        'classes_': model.classes_.tolist(),
        'coef_': model.coef_.tolist(),
        'intercept_': model.intercept_.tolist(),
        'n_iter_': model.n_iter_.tolist(),
        'params': model.get_params()
    }

    return serialized_model


def deserialize_logistic_regression_clf(model_dict, version_out):
    model = LogisticRegression(model_dict['params'])

    model.classes_ = np.array(model_dict['classes_'])
    model.coef_ = np.array(model_dict['coef_'])
    model.intercept_ = np.array(model_dict['intercept_'])
    model.n_iter_ = np.array(model_dict['intercept_'])

    return model