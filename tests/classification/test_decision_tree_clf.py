from sklearn.tree import DecisionTreeClassifier
from sklearn_migrator.classification.decision_tree_clf import serialize_decision_tree_clf
from sklearn_migrator.classification.decision_tree_clf import deserialize_decision_tree_clf
import sklearn

def test_decision_tree_clf():
    X = [[0], [1], [2], [3]]
    y = [0, 1, 1, 0]
    
    model = DecisionTreeClassifier()
    model.fit(X, y)

    version = sklearn.__version__
    result = serialize_decision_tree_clf(model, version_in=version)
    new_model = deserialize_decision_tree_clf(result, version_out=version)

    #--------------------------------------------------

    assert isinstance(result, dict)

    assert 'version_sklearn_in' in result

    assert isinstance(new_model, DecisionTreeClassifier)

    #--------------------------------------------------

    y_pred = model.predict_proba(X)
    y_pred_new = new_model.predict_proba(X)

    threshold = 0.001

    assert (abs(y_pred - y_pred_new).max(axis = 1).max() <= threshold)

    #--------------------------------------------------