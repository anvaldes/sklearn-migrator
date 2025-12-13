from sklearn.ensemble import GradientBoostingClassifier
from sklearn_migrator.classification.gradient_boosting_clf import serialize_gradient_boosting_clf
from sklearn_migrator.classification.gradient_boosting_clf import deserialize_gradient_boosting_clf
import sklearn

def test_gradient_boosting_clf():
    X = [[0], [1], [2], [3]]
    y = [0, 1, 0, 1]
    
    model = GradientBoostingClassifier(n_estimators=1, max_depth=1)
    model.fit(X, y)

    version = sklearn.__version__
    result = serialize_gradient_boosting_clf(model, version_in=version)
    new_model = deserialize_gradient_boosting_clf(result, version_out=version)

    #--------------------------------------------------

    assert isinstance(result, dict)

    assert 'version_sklearn_in' in result

    assert isinstance(new_model, GradientBoostingClassifier)

    #--------------------------------------------------

    y_pred = model.predict_proba(X)
    y_pred_new = new_model.predict_proba(X)

    threshold = 0.001

    assert (abs(y_pred - y_pred_new).max(axis = 1).max() <= threshold)

    #--------------------------------------------------