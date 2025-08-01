from sklearn.linear_model import LinearRegression
from sklearn_migrator.regression.linear_regression_reg import serialize_linear_regression_reg
from sklearn_migrator.regression.linear_regression_reg import deserialize_linear_regression_reg
import sklearn

def test_linear_regression_reg():
    X = [[0], [1], [2], [3]]
    y = [0, 1, 1, 0]
    
    model = LinearRegression()
    model.fit(X, y)

    version = sklearn.__version__
    result = serialize_linear_regression_reg(model, version_in=version)
    new_model = deserialize_linear_regression_reg(result, version_out=version)

    assert isinstance(result, dict)

    assert 'version_sklearn_in' in result

    assert isinstance(new_model, LinearRegression)