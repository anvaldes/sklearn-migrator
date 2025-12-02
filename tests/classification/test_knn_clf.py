from sklearn.neighbors import KNeighborsClassifier
from sklearn_migrator.classification.knn_clf import serialize_knn_clf
from sklearn_migrator.classification.knn_clf import deserialize_knn_clf
import sklearn

def test_knn_clf():
    X = [[0], [1], [2], [3]]
    y = [0, 1, 2, 3]
    
    model = KNeighborsClassifier()
    model.fit(X, y)

    version = sklearn.__version__
    result = serialize_knn_clf(model, version_in=version)
    new_model = deserialize_knn_clf(result, version_out=version)

    assert isinstance(result, dict)

    assert 'version_sklearn_in' in result

    assert isinstance(new_model, KNeighborsClassifier)