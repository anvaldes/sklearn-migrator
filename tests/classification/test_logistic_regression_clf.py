from sklearn.linear_model import LogisticRegression
from sklearn_migrator.classification.logistic_regression_clf import serialize_logistic_regression_clf
from sklearn_migrator.classification.logistic_regression_clf import deserialize_logistic_regression_clf
import sklearn

def test_logistic_regression_clf():
    X = [[0], [1], [2], [3]]
    y = [0, 1, 1, 0]
    
    model = LogisticRegression()
    model.fit(X, y)

    version = sklearn.__version__
    result = serialize_logistic_regression_clf(model, version_in=version)
    new_model = deserialize_logistic_regression_clf(result, version_out=version)

    #--------------------------------------------------

    assert isinstance(result, dict)

    assert 'version_sklearn_in' in result

    assert isinstance(new_model, LogisticRegression)

    #--------------------------------------------------

    y_pred = model.predict_proba(X)
    y_pred_new = new_model.predict_proba(X)

    threshold = 0.001

    assert (abs(y_pred - y_pred_new).max(axis = 1).max() <= threshold)

    #--------------------------------------------------