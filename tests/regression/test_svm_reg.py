from sklearn.svm import SVR
from sklearn_migrator.regression.svm_reg import serialize_svr
from sklearn_migrator.regression.svm_reg import deserialize_svr
import sklearn

def test_random_forest_reg():
    X = [[0], [1], [2], [3]]
    y = [0, 1, 2, 3]
    
    model = SVR()
    model.fit(X, y)

    version = sklearn.__version__
    result = serialize_svr(model, version_in=version)
    new_model = deserialize_svr(result, version_out=version)

    assert isinstance(result, dict)

    assert 'version_sklearn_in' in result