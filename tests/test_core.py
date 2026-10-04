import json
import importlib
import numpy as np
import pytest
import sklearn
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn_migrator import serialize, deserialize, supported_models
from sklearn_migrator.core import _REGISTRY

X = np.array([[0.0, 1.0], [1.0, 0.5], [2.0, 2.5], [3.0, 1.5], [4.0, 4.5], [5.0, 3.5]])
y_clf = np.array([0, 1, 0, 1, 0, 1])
y_reg = np.array([0.5, 1.0, 2.5, 3.0, 4.5, 5.0])

# (model, target, method used to compare the original and the migrated model)
MODELS = [
    (DecisionTreeClassifier(random_state=0), y_clf, 'predict_proba'),
    (RandomForestClassifier(n_estimators=5, random_state=0), y_clf, 'predict_proba'),
    (GradientBoostingClassifier(n_estimators=5, random_state=0), y_clf, 'predict_proba'),
    (LogisticRegression(), y_clf, 'predict_proba'),
    (KNeighborsClassifier(n_neighbors=2), y_clf, 'predict_proba'),
    (SVC(), y_clf, 'predict'),
    (MLPClassifier(max_iter=50, random_state=0), y_clf, 'predict_proba'),
    (DecisionTreeRegressor(random_state=0), y_reg, 'predict'),
    (RandomForestRegressor(n_estimators=5, random_state=0), y_reg, 'predict'),
    (GradientBoostingRegressor(n_estimators=5, random_state=0), y_reg, 'predict'),
    (LinearRegression(), y_reg, 'predict'),
    (Ridge(), y_reg, 'predict'),
    (Lasso(), y_reg, 'predict'),
    (KNeighborsRegressor(n_neighbors=2), y_reg, 'predict'),
    (SVR(), y_reg, 'predict'),
    (AdaBoostRegressor(n_estimators=5, random_state=0), y_reg, 'predict'),
    (MLPRegressor(max_iter=50, random_state=0), y_reg, 'predict'),
    (KMeans(n_clusters=2, n_init=10, random_state=0), None, 'predict'),
    (MiniBatchKMeans(n_clusters=2, n_init=3, random_state=0), None, 'predict'),
    (AgglomerativeClustering(n_clusters=2), None, 'labels_'),
    (PCA(n_components=2), None, 'transform'),
]

MODEL_IDS = [type(model).__name__ for model, _, _ in MODELS]


def _fit(model, y):
    return model.fit(X) if y is None else model.fit(X, y)


def _output(model, method):
    attr = getattr(model, method)
    return np.asarray(attr(X) if callable(attr) else attr, dtype=float)


def test_every_supported_model_is_tested():
    assert sorted(MODEL_IDS) == supported_models()
    assert len(supported_models()) == 21


@pytest.mark.parametrize("model, y, method", MODELS, ids=MODEL_IDS)
def test_roundtrip_detects_model(model, y, method):
    """
    serialize/deserialize pick the right functions without being told the model.
    """
    model = _fit(model, y)

    data = serialize(model)

    assert data['model_type'] == type(model).__name__
    assert data['version_sklearn_in'] == sklearn.__version__

    # The dictionary must survive a JSON round trip, as it does between environments
    new_model = deserialize(json.loads(json.dumps(data)))

    assert np.abs(_output(model, method) - _output(new_model, method)).max() <= 1e-2


@pytest.mark.parametrize("model, y, method", MODELS, ids=MODEL_IDS)
def test_matches_model_specific_functions(model, y, method):
    """
    The unified functions return the same as the model-specific ones, which
    remain available, and can deserialize their output given the model type.
    """
    model = _fit(model, y)
    model_type = type(model).__name__
    version = sklearn.__version__

    module_name, serialize_name, deserialize_name = _REGISTRY[model_type]
    module = importlib.import_module('sklearn_migrator.' + module_name)

    legacy_data = getattr(module, serialize_name)(model, version)
    assert 'model_type' not in legacy_data

    data = serialize(model, version_in=version)
    data.pop('model_type')
    assert json.dumps(data) == json.dumps(legacy_data)

    legacy_model = getattr(module, deserialize_name)(json.loads(json.dumps(legacy_data)), version)
    new_model = deserialize(json.loads(json.dumps(legacy_data)), version_out=version, model_type=model_type)

    assert type(new_model) is type(legacy_model)
    assert np.abs(_output(legacy_model, method) - _output(new_model, method)).max() <= 1e-2


def test_explicit_version_is_stored():
    model = LinearRegression().fit(X, y_reg)

    assert serialize(model, version_in='1.2.0')['version_sklearn_in'] == '1.2.0'
    assert serialize(model, '1.2.0')['version_sklearn_in'] == '1.2.0'


def test_serialize_unsupported_sklearn_model():
    model = ExtraTreesClassifier(n_estimators=2).fit(X, y_clf)

    with pytest.raises(TypeError, match="ExtraTreesClassifier.*not supported"):
        serialize(model)


def test_serialize_non_sklearn_object():
    class KMeans:
        pass

    with pytest.raises(TypeError, match="not supported"):
        serialize(KMeans())

    with pytest.raises(TypeError, match="not supported"):
        serialize({'a': 1})


def test_serialize_subclass_is_not_treated_as_parent():
    class MyForest(RandomForestRegressor):
        pass

    with pytest.raises(TypeError, match="MyForest.*not supported"):
        serialize(MyForest(n_estimators=2).fit(X, y_reg))


def test_deserialize_without_model_type():
    data = serialize(LinearRegression().fit(X, y_reg))
    data.pop('model_type')

    with pytest.raises(ValueError, match="model_type"):
        deserialize(data)

    assert isinstance(deserialize(data, model_type='LinearRegression'), LinearRegression)


def test_deserialize_unknown_model_type():
    data = serialize(LinearRegression().fit(X, y_reg))
    data['model_type'] = 'ExtraTreesClassifier'

    with pytest.raises(ValueError, match="ExtraTreesClassifier.*not supported"):
        deserialize(data)
