from sklearn.linear_model import LogisticRegression
from sklearn_migrator.classification.logistic_regression_clf import serialize_logistic_regression_clf
from sklearn_migrator.classification.logistic_regression_clf import deserialize_logistic_regression_clf
import sklearn

def test_decision_tree_clf():
    X = [[0], [1], [2], [3]]
    y = [0, 1, 1, 0]
    
    model = LogisticRegression()
    model.fit(X, y)

    version = sklearn.__version__
    result = serialize_logistic_regression_clf(model, version_in=version)
    new_model = deserialize_logistic_regression_clf(result, version_out=version)

    assert isinstance(result, dict)

    assert 'classes_' in result
    assert 'coef_' in result
    assert 'intercept_' in result

    assert isinstance(new_model, LogisticRegression)