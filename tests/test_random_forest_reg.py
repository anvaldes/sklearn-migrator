from sklearn.ensemble import RandomForestRegressor
from sklearn_migrator.regression.random_forest_reg import serialize_random_forest_reg
from sklearn_migrator.regression.random_forest_reg import deserialize_random_forest_reg
import sklearn

def test_random_forest_reg():
    X = [[0], [1], [2], [3]]
    y = [0, 1, 2, 3]
    
    model = RandomForestRegressor()
    model.fit(X, y)

    version = sklearn.__version__
    result = serialize_random_forest_reg(model, version_in=version)
    new_model = deserialize_random_forest_reg(result, version_out=version)

    assert isinstance(result, dict)

    assert 'n_outputs_' in result['other_params']
    assert 'version_sklearn_in' in result

    assert isinstance(new_model, RandomForestRegressor)