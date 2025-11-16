from sklearn.linear_model import Lasso
from sklearn_migrator.regression.lasso_reg import serialize_lasso_reg
from sklearn_migrator.regression.lasso_reg import deserialize_lasso_reg
import sklearn

def test_lasso_reg():
    X = [[0], [1], [2], [3]]
    y = [0, 1, 2, 3]
    
    model = Lasso()
    model.fit(X, y)

    version = sklearn.__version__
    result = serialize_lasso_reg(model, version_in=version)
    new_model = deserialize_lasso_reg(result, version_out=version)

    assert isinstance(result, dict)

    assert 'version_sklearn_in' in result

    assert isinstance(new_model, Lasso)