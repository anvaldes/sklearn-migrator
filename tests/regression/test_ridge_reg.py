from sklearn.linear_model import Ridge
from sklearn_migrator.regression.ridge_reg import serialize_ridge_reg
from sklearn_migrator.regression.ridge_reg import deserialize_ridge_reg
import sklearn

def test_ridge_reg():
    X = [[0], [1], [2], [3]]
    y = [0, 1, 2, 3]
    
    model = Ridge()
    model.fit(X, y)

    version = sklearn.__version__
    result = serialize_ridge_reg(model, version_in=version)
    new_model = deserialize_ridge_reg(result, version_out=version)

    assert isinstance(result, dict)

    assert 'version_sklearn_in' in result

    assert isinstance(new_model, Ridge)