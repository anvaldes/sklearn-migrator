from sklearn.neural_network import MLPClassifier
from sklearn_migrator.classification.mlp_reg import serialize_mlp_clf
from sklearn_migrator.classification.mlp_reg import deserialize_mlp_clf
import sklearn

def test_mlp_clf():
    X = [[0], [1], [2], [3]]
    y = [0, 1, 2, 3]
    
    model = MLPClassifier()
    model.fit(X, y)

    version = sklearn.__version__
    result = serialize_mlp_clf(model, version_in=version)
    new_model = deserialize_mlp_clf(result, version_out=version)

    assert isinstance(result, dict)

    assert 'version_sklearn_in' in result

    assert isinstance(new_model, MLPClassifier)