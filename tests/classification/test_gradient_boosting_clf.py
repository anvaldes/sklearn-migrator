from sklearn.ensemble import GradientBoostingClassifier
from sklearn_migrator.classification.gradient_boosting_clf import serialize_gradient_boosting_clf
from sklearn_migrator.classification.gradient_boosting_clf import deserialize_gradient_boosting_clf
import sklearn

def test_gradient_boosting_clf():
    X = [[0], [1], [2], [3]]
    y = [0, 1, 2, 3]
    
    model = GradientBoostingClassifier()
    model.fit(X, y)

    version = sklearn.__version__
    result = serialize_gradient_boosting_clf(model, version_in=version)
    new_model = deserialize_gradient_boosting_clf(result, version_out=version)

    assert isinstance(result, dict)

    assert 'version_sklearn_in' in result

    assert isinstance(new_model, GradientBoostingClassifier)