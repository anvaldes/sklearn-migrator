from sklearn.neighbors import KNeighborsRegressor
from sklearn_migrator.regression.knn_reg import serialize_knn_reg
from sklearn_migrator.regression.knn_reg import deserialize_knn_reg
import sklearn

def test_mlp_reg():
    X = [[0], [1], [2], [3]]
    y = [0, 1, 2, 3]
    
    model = KNeighborsRegressor()
    model.fit(X, y)

    version = sklearn.__version__
    result = serialize_knn_reg(model, version_in=version)
    new_model = deserialize_knn_reg(result, version_out=version)

    assert isinstance(result, dict)

    assert 'version_sklearn_in' in result

    assert isinstance(new_model, KNeighborsRegressor)