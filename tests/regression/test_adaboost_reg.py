from sklearn.ensemble import AdaBoostRegressor
from sklearn_migrator.regression.adaboost_reg import serialize_adaboost_reg
from sklearn_migrator.regression.adaboost_reg import deserialize_adaboost_reg
import sklearn

def test_adaboost_reg():
    X = [[0], [1], [2], [3]]
    y = [0, 1, 2, 3]
    
    model = AdaBoostRegressor()
    model.fit(X, y)

    version = sklearn.__version__
    result = serialize_adaboost_reg(model, version_in=version)
    new_model = deserialize_adaboost_reg(result, version_out=version)

    assert isinstance(result, dict)

    assert 'version_sklearn_in' in result

    assert isinstance(new_model, AdaBoostRegressor)