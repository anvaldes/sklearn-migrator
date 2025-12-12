from sklearn.svm import SVC
from sklearn_migrator.classification.svm_clf import serialize_svc
from sklearn_migrator.classification.svm_clf import deserialize_svc
import sklearn

def test_svm_clf():
    X = [[0], [1], [2], [3]]
    y = [0, 1, 0, 1]
    
    model = SVC()
    model.fit(X, y)

    version = sklearn.__version__
    result = serialize_svc(model, version_in=version)
    new_model = deserialize_svc(result, version_out=version)

    #--------------------------------------------------

    assert isinstance(result, dict)

    assert 'version_sklearn_in' in result

    #--------------------------------------------------

    y_pred = model.predict(X)
    y_pred_new = new_model.predict(X)

    threshold = 0.001

    assert (abs(y_pred - y_pred_new).max() <= threshold)

    #--------------------------------------------------