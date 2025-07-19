import numpy as np
from sklearn.linear_model import LogisticRegression

def serialize_logistic_regression_clf(model, version_in):

    metadata = {
        'meta': 'lr',
        'classes_': model.classes_.tolist(),
        'coef_': model.coef_.tolist(),
        'intercept_': model.intercept_.tolist(),
        'n_iter_': model.n_iter_.tolist(),
        'params': model.get_params(),
        'version_sklearn_in': version_in
    }

    return metadata


def deserialize_logistic_regression_clf(data, version_out):

    model = LogisticRegression(data['params'])

    model.classes_ = np.array(data['classes_'])
    model.coef_ = np.array(data['coef_'])
    model.intercept_ = np.array(data['intercept_'])
    model.n_iter_ = np.array(data['intercept_'])

    return model