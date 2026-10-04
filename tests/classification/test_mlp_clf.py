from sklearn.neural_network import MLPClassifier
from sklearn_migrator import serialize, deserialize
import sklearn

def test_mlp_clf():
    X = [[0], [1], [2], [3]]
    y = [0, 1, 2, 3]
    
    model = MLPClassifier()
    model.fit(X, y)

    version = sklearn.__version__
    result = serialize(model, version_in=version)
    new_model = deserialize(result, version_out=version)

    #--------------------------------------------------

    assert isinstance(result, dict)

    assert 'version_sklearn_in' in result

    assert isinstance(new_model, MLPClassifier)

    #--------------------------------------------------

    y_pred = model.predict_proba(X)
    y_pred_new = new_model.predict_proba(X)

    threshold = 1e-2

    assert (abs(y_pred - y_pred_new).max(axis = 1).max() <= threshold)

    #--------------------------------------------------