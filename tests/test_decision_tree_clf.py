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

    assert isinstance(result, dict)

    assert 'n_features_in' in result
    assert 'n_features' in result
    assert 'n_outputs' in result
    assert 'n_classes' in result
    assert 'serialized_tree' in result
    assert 'version_sklearn_in' in result

    assert result['serialized_tree']['max_depth'] > 0
    assert isinstance(result['serialized_tree']['nodes'], list)

    assert isinstance(new_model, DecisionTreeClassifier)