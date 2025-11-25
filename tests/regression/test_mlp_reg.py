from sklearn.neural_network import MLPRegressor
from sklearn_migrator.regression.mlp_reg import serialize_mlp_reg
from sklearn_migrator.regression.mlp_reg import deserialize_mlp_reg
import sklearn

def test_mlp_reg():
    X = [[0], [1], [2], [3]]
    y = [0, 1, 2, 3]
    
    model = MLPRegressor()
    model.fit(X, y)

    version = sklearn.__version__
    result = serialize_mlp_reg(model, version_in=version)
    new_model = deserialize_mlp_reg(result, version_out=version)

    assert isinstance(result, dict)

    assert 'version_sklearn_in' in result

    assert isinstance(new_model, MLPRegressor)