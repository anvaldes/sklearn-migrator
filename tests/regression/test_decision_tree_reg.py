from sklearn.tree import DecisionTreeRegressor
from sklearn_migrator.regression.decision_tree_reg import serialize_decision_tree_reg
from sklearn_migrator.regression.decision_tree_reg import deserialize_decision_tree_reg
import sklearn

def test_decision_tree_reg():
    X = [[0], [1], [2], [3]]
    y = [0, 1, 2, 3]
    
    model = DecisionTreeRegressor()
    model.fit(X, y)

    version = sklearn.__version__
    result = serialize_decision_tree_reg(model, version_in=version)
    new_model = deserialize_decision_tree_reg(result, version_out=version)

    #--------------------------------------------------

    assert isinstance(result, dict)

    assert 'version_sklearn_in' in result

    assert isinstance(new_model, DecisionTreeRegressor)

    #--------------------------------------------------

    y_pred = model.predict(X)
    y_pred_new = new_model.predict(X)

    threshold = 0.001

    assert (abs(y_pred - y_pred_new).max() <= threshold)

    #--------------------------------------------------