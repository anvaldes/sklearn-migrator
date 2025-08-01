from sklearn.ensemble import RandomForestClassifier
from sklearn_migrator.classification.random_forest_clf import serialize_random_forest_clf
from sklearn_migrator.classification.random_forest_clf import deserialize_random_forest_clf
import sklearn

def test_random_forest_clf():
    X = [[0], [1], [2], [3]]
    y = [0, 1, 2, 3]
    
    model = RandomForestClassifier()
    model.fit(X, y)

    version = sklearn.__version__
    result = serialize_random_forest_clf(model, version_in=version)
    new_model = deserialize_random_forest_clf(result, version_out=version)

    assert isinstance(result, dict)

    assert 'version_sklearn_in' in result

    assert isinstance(new_model, RandomForestClassifier)