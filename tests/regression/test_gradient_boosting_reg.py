from sklearn.ensemble import GradientBoostingRegressor
from sklearn_migrator.regression.gradient_boosting_reg import serialize_gradient_boosting_reg
from sklearn_migrator.regression.gradient_boosting_reg import deserialize_gradient_boosting_reg
import sklearn

def test_gradient_boosting_reg():
    X = [[0], [1], [2], [3]]
    y = [0, 1, 2, 3]
    
    model = GradientBoostingRegressor()
    model.fit(X, y)

    version = sklearn.__version__
    result = serialize_gradient_boosting_reg(model, version_in=version)
    new_model = deserialize_gradient_boosting_reg(result, version_out=version)

    assert isinstance(result, dict)

    assert 'version_sklearn_in' in result

    assert isinstance(new_model, GradientBoostingRegressor)