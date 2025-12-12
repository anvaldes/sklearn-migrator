from sklearn.neighbors import KNeighborsRegressor
from sklearn_migrator.regression.knn_reg import serialize_knn_reg
from sklearn_migrator.regression.knn_reg import deserialize_knn_reg
import sklearn

def test_mlp_reg():
    X = [[0], [1], [2], [3]]
    y = [0, 1, 2, 3]
    
    model = KNeighborsRegressor(n_neighbors=3)
    model.fit(X, y)

    version = sklearn.__version__
    result = serialize_knn_reg(model, version_in=version)
    new_model = deserialize_knn_reg(result, version_out=version)

    #--------------------------------------------------

    assert isinstance(result, dict)

    assert 'version_sklearn_in' in result

    assert isinstance(new_model, KNeighborsRegressor)

    #--------------------------------------------------

    y_pred = model.predict(X)
    y_pred_new = new_model.predict(X)

    threshold = 0.001

    assert (abs(y_pred - y_pred_new).max() <= threshold)

    #--------------------------------------------------