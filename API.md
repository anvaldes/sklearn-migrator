# sklearn-migrator API Reference

This document covers every public `serialize` and `deserialize` function in the library.
The general pattern is always the same: call `serialize_<model>(model, version_in)` on the
source environment, persist or transfer the resulting dictionary (e.g. as JSON), then call
`deserialize_<model>(data, version_out)` on the target environment to recover a working model.

Supported sklearn versions: **0.21.3 – 1.7.2**

---

## Table of Contents

- [Classification](#classification)
  - [Logistic Regression](#logistic-regression)
  - [Decision Tree Classifier](#decision-tree-classifier)
  - [Random Forest Classifier](#random-forest-classifier)
  - [Gradient Boosting Classifier](#gradient-boosting-classifier)
  - [K-Nearest Neighbors Classifier](#k-nearest-neighbors-classifier)
  - [Support Vector Classifier (SVC)](#support-vector-classifier-svc)
  - [MLP Classifier](#mlp-classifier)
- [Regression](#regression)
  - [Linear Regression](#linear-regression)
  - [Ridge Regression](#ridge-regression)
  - [Lasso Regression](#lasso-regression)
  - [Decision Tree Regressor](#decision-tree-regressor)
  - [Random Forest Regressor](#random-forest-regressor)
  - [Gradient Boosting Regressor](#gradient-boosting-regressor)
  - [AdaBoost Regressor](#adaboost-regressor)
  - [MLP Regressor](#mlp-regressor)
  - [Support Vector Regressor (SVR)](#support-vector-regressor-svr)
  - [K-Nearest Neighbors Regressor](#k-nearest-neighbors-regressor)
- [Clustering](#clustering)
  - [KMeans](#kmeans)
  - [MiniBatchKMeans](#minibatchkmeans)
  - [Agglomerative Clustering](#agglomerative-clustering)
- [Dimensionality Reduction](#dimensionality-reduction)
  - [PCA](#pca)
- [Common Patterns](#common-patterns)

---

## Classification

### Logistic Regression

```python
from sklearn_migrator.classification.logistic_regression_clf import (
    serialize_logistic_regression_clf,
    deserialize_logistic_regression_clf,
)
```

---

#### `serialize_logistic_regression_clf`

```python
serialize_logistic_regression_clf(model: LogisticRegression, version_in: str) -> dict
```

Converts a fitted `LogisticRegression` into a JSON-compatible dictionary that captures all coefficients, hyperparameters, and version-specific attributes.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `model` | `LogisticRegression` | A fitted scikit-learn `LogisticRegression` instance. |
| `version_in` | `str` | The sklearn version used to train the model (e.g. `'1.2.0'`). |

**Returns**

| Type | Description |
|------|-------------|
| `dict` | A JSON-serializable dictionary containing coefficients, intercept, iteration counts, hyperparameters, and version metadata. |

**Example**

```python
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn_migrator.classification.logistic_regression_clf import serialize_logistic_regression_clf

model = LogisticRegression().fit(X_train, y_train)
data = serialize_logistic_regression_clf(model, sklearn.__version__)

import json
with open("logistic_regression.json", "w") as f:
    json.dump(data, f)
```

---

#### `deserialize_logistic_regression_clf`

```python
deserialize_logistic_regression_clf(data: dict, version_out: str) -> LogisticRegression
```

Reconstructs a `LogisticRegression` from a serialized dictionary, restoring all learned attributes for the target sklearn version.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `data` | `dict` | Dictionary produced by `serialize_logistic_regression_clf`. |
| `version_out` | `str` | The sklearn version of the target environment (e.g. `'1.7.0'`). |

**Returns**

| Type | Description |
|------|-------------|
| `LogisticRegression` | A fully reconstructed scikit-learn `LogisticRegression` ready to call `.predict()` and `.predict_proba()`. |

**Example**

```python
import json
import sklearn
from sklearn_migrator.classification.logistic_regression_clf import deserialize_logistic_regression_clf

with open("logistic_regression.json") as f:
    data = json.load(f)

model = deserialize_logistic_regression_clf(data, sklearn.__version__)
predictions = model.predict(X_test)
```

---

### Decision Tree Classifier

```python
from sklearn_migrator.classification.decision_tree_clf import (
    serialize_decision_tree_clf,
    deserialize_decision_tree_clf,
)
```

---

#### `serialize_decision_tree_clf`

```python
serialize_decision_tree_clf(model: DecisionTreeClassifier, version_in: str) -> dict
```

Converts a fitted `DecisionTreeClassifier` into a JSON-compatible dictionary, including the full internal tree structure serialized as node arrays.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `model` | `DecisionTreeClassifier` | A fitted scikit-learn `DecisionTreeClassifier` instance. |
| `version_in` | `str` | The sklearn version used to train the model (e.g. `'1.2.0'`). |

**Returns**

| Type | Description |
|------|-------------|
| `dict` | A JSON-serializable dictionary containing the tree's node structure, split values, class counts, and all hyperparameters. |

**Example**

```python
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn_migrator.classification.decision_tree_clf import serialize_decision_tree_clf

model = DecisionTreeClassifier().fit(X_train, y_train)
data = serialize_decision_tree_clf(model, sklearn.__version__)
```

---

#### `deserialize_decision_tree_clf`

```python
deserialize_decision_tree_clf(data: dict, version_out: str) -> DecisionTreeClassifier
```

Reconstructs a `DecisionTreeClassifier` from a serialized dictionary, handling node-structure differences between sklearn versions automatically.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `data` | `dict` | Dictionary produced by `serialize_decision_tree_clf`. |
| `version_out` | `str` | The sklearn version of the target environment (e.g. `'1.7.0'`). |

**Returns**

| Type | Description |
|------|-------------|
| `DecisionTreeClassifier` | A fully reconstructed scikit-learn `DecisionTreeClassifier` ready to call `.predict()`. |

**Example**

```python
import sklearn
from sklearn_migrator.classification.decision_tree_clf import deserialize_decision_tree_clf

model = deserialize_decision_tree_clf(data, sklearn.__version__)
predictions = model.predict(X_test)
```

---

### Random Forest Classifier

```python
from sklearn_migrator.classification.random_forest_clf import (
    serialize_random_forest_clf,
    deserialize_random_forest_clf,
)
```

---

#### `serialize_random_forest_clf`

```python
serialize_random_forest_clf(model: RandomForestClassifier, version_in: str) -> dict
```

Converts a fitted `RandomForestClassifier` into a JSON-compatible dictionary by serializing each constituent decision tree individually.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `model` | `RandomForestClassifier` | A fitted scikit-learn `RandomForestClassifier` instance. |
| `version_in` | `str` | The sklearn version used to train the model (e.g. `'1.2.0'`). |

**Returns**

| Type | Description |
|------|-------------|
| `dict` | A JSON-serializable dictionary containing the serialized estimator list, ensemble hyperparameters, and version metadata. |

**Example**

```python
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn_migrator.classification.random_forest_clf import serialize_random_forest_clf

model = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
data = serialize_random_forest_clf(model, sklearn.__version__)
```

---

#### `deserialize_random_forest_clf`

```python
deserialize_random_forest_clf(data: dict, version_out: str) -> RandomForestClassifier
```

Reconstructs a `RandomForestClassifier` from a serialized dictionary, rebuilding all constituent trees for the target sklearn version.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `data` | `dict` | Dictionary produced by `serialize_random_forest_clf`. |
| `version_out` | `str` | The sklearn version of the target environment (e.g. `'1.7.0'`). |

**Returns**

| Type | Description |
|------|-------------|
| `RandomForestClassifier` | A fully reconstructed scikit-learn `RandomForestClassifier` ready to call `.predict()` and `.predict_proba()`. |

**Example**

```python
import sklearn
from sklearn_migrator.classification.random_forest_clf import deserialize_random_forest_clf

model = deserialize_random_forest_clf(data, sklearn.__version__)
predictions = model.predict(X_test)
```

---

### Gradient Boosting Classifier

```python
from sklearn_migrator.classification.gradient_boosting_clf import (
    serialize_gradient_boosting_clf,
    deserialize_gradient_boosting_clf,
)
```

---

#### `serialize_gradient_boosting_clf`

```python
serialize_gradient_boosting_clf(model: GradientBoostingClassifier, version_in: str) -> dict
```

Converts a fitted `GradientBoostingClassifier` into a JSON-compatible dictionary, including each boosting stage's regression tree and the initial estimator state.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `model` | `GradientBoostingClassifier` | A fitted scikit-learn `GradientBoostingClassifier` instance. |
| `version_in` | `str` | The sklearn version used to train the model (e.g. `'1.2.0'`). |

**Returns**

| Type | Description |
|------|-------------|
| `dict` | A JSON-serializable dictionary containing all boosting stage trees, the dummy classifier state, loss function name, training scores, and hyperparameters. |

**Example**

```python
import sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn_migrator.classification.gradient_boosting_clf import serialize_gradient_boosting_clf

model = GradientBoostingClassifier().fit(X_train, y_train)
data = serialize_gradient_boosting_clf(model, sklearn.__version__)
```

---

#### `deserialize_gradient_boosting_clf`

```python
deserialize_gradient_boosting_clf(data: dict, version_out: str) -> GradientBoostingClassifier
```

Reconstructs a `GradientBoostingClassifier` from a serialized dictionary, rewiring the loss function and initial estimator to match the target sklearn version.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `data` | `dict` | Dictionary produced by `serialize_gradient_boosting_clf`. |
| `version_out` | `str` | The sklearn version of the target environment (e.g. `'1.7.0'`). |

**Returns**

| Type | Description |
|------|-------------|
| `GradientBoostingClassifier` | A fully reconstructed scikit-learn `GradientBoostingClassifier` ready to call `.predict()` and `.predict_proba()`. |

**Example**

```python
import sklearn
from sklearn_migrator.classification.gradient_boosting_clf import deserialize_gradient_boosting_clf

model = deserialize_gradient_boosting_clf(data, sklearn.__version__)
predictions = model.predict(X_test)
```

---

### K-Nearest Neighbors Classifier

```python
from sklearn_migrator.classification.knn_clf import (
    serialize_knn_clf,
    deserialize_knn_clf,
)
```

---

#### `serialize_knn_clf`

```python
serialize_knn_clf(model: KNeighborsClassifier, version_in: str) -> dict
```

Converts a fitted `KNeighborsClassifier` into a JSON-compatible dictionary by capturing the full training data and hyperparameters.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `model` | `KNeighborsClassifier` | A fitted scikit-learn `KNeighborsClassifier` instance. |
| `version_in` | `str` | The sklearn version used to train the model (e.g. `'1.2.0'`). |

**Returns**

| Type | Description |
|------|-------------|
| `dict` | A JSON-serializable dictionary containing the training data matrix, target labels, and all hyperparameters. |

> **Note:** Because KNN is lazy (no training phase), the serialized dictionary stores the full training set. Dictionary size scales with the number of training samples.

**Example**

```python
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn_migrator.classification.knn_clf import serialize_knn_clf

model = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
data = serialize_knn_clf(model, sklearn.__version__)
```

---

#### `deserialize_knn_clf`

```python
deserialize_knn_clf(data: dict, version_out: str) -> KNeighborsClassifier
```

Reconstructs a `KNeighborsClassifier` by re-fitting it on the stored training data using the original hyperparameters.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `data` | `dict` | Dictionary produced by `serialize_knn_clf`. |
| `version_out` | `str` | The sklearn version of the target environment (e.g. `'1.7.0'`). |

**Returns**

| Type | Description |
|------|-------------|
| `KNeighborsClassifier` | A fully reconstructed scikit-learn `KNeighborsClassifier` ready to call `.predict()`. |

**Example**

```python
import sklearn
from sklearn_migrator.classification.knn_clf import deserialize_knn_clf

model = deserialize_knn_clf(data, sklearn.__version__)
predictions = model.predict(X_test)
```

---

### Support Vector Classifier (SVC)

```python
from sklearn_migrator.classification.svm_clf import (
    serialize_svc,
    deserialize_svc,
)
```

> **Important:** `deserialize_svc` returns a `Migrated_SVC` object, not a native `sklearn.svm.SVC`. `Migrated_SVC` is a pure-NumPy reimplementation that avoids sklearn's compiled C extensions, making it stable across all supported versions. It exposes `.predict()`, `.decision_function()`, and `.predict_proba()` (when the original model was trained with `probability=True`). It supports only **binary classification**.

---

#### `serialize_svc`

```python
serialize_svc(model: SVC, version_in: str) -> dict
```

Converts a fitted binary `SVC` into a JSON-compatible dictionary capturing the support vectors, dual coefficients, intercept, and kernel parameters.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `model` | `SVC` | A fitted scikit-learn `SVC` instance (binary classification only). |
| `version_in` | `str` | The sklearn version used to train the model (e.g. `'1.2.0'`). |

**Returns**

| Type | Description |
|------|-------------|
| `dict` | A JSON-serializable dictionary containing support vectors, dual coefficients, intercept, kernel parameters, Platt scaling coefficients (if available), and version metadata. |

**Example**

```python
import sklearn
from sklearn.svm import SVC
from sklearn_migrator.classification.svm_clf import serialize_svc

model = SVC(kernel='rbf', probability=True).fit(X_train, y_train)
data = serialize_svc(model, sklearn.__version__)
```

---

#### `deserialize_svc`

```python
deserialize_svc(data: dict, version_out: str) -> Migrated_SVC
```

Reconstructs a `Migrated_SVC` from a serialized dictionary, providing a version-independent prediction interface.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `data` | `dict` | Dictionary produced by `serialize_svc`. |
| `version_out` | `str` | The sklearn version of the target environment (e.g. `'1.7.0'`). |

**Returns**

| Type | Description |
|------|-------------|
| `Migrated_SVC` | A pure-NumPy binary classifier with `.predict()`, `.decision_function()`, and (optionally) `.predict_proba()`. |

**Example**

```python
import sklearn
from sklearn_migrator.classification.svm_clf import deserialize_svc

model = deserialize_svc(data, sklearn.__version__)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)  # only if probability=True at training time
```

---

### MLP Classifier

```python
from sklearn_migrator.classification.mlp_clf import (
    serialize_mlp_clf,
    deserialize_mlp_clf,
)
```

---

#### `serialize_mlp_clf`

```python
serialize_mlp_clf(model: MLPClassifier, version_in: str) -> dict
```

Converts a fitted `MLPClassifier` into a JSON-compatible dictionary capturing all layer weights, biases, and training metadata.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `model` | `MLPClassifier` | A fitted scikit-learn `MLPClassifier` instance. |
| `version_in` | `str` | The sklearn version used to train the model (e.g. `'1.2.0'`). |

**Returns**

| Type | Description |
|------|-------------|
| `dict` | A JSON-serializable dictionary containing all weight matrices (`coefs_`), bias vectors (`intercepts_`), training loss, layer counts, output activation, and hyperparameters. |

**Example**

```python
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn_migrator.classification.mlp_clf import serialize_mlp_clf

model = MLPClassifier(hidden_layer_sizes=(100,)).fit(X_train, y_train)
data = serialize_mlp_clf(model, sklearn.__version__)
```

---

#### `deserialize_mlp_clf`

```python
deserialize_mlp_clf(data: dict, version_out: str) -> MLPClassifier
```

Reconstructs a `MLPClassifier` from a serialized dictionary, restoring all weights, biases, and network architecture.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `data` | `dict` | Dictionary produced by `serialize_mlp_clf`. |
| `version_out` | `str` | The sklearn version of the target environment (e.g. `'1.7.0'`). |

**Returns**

| Type | Description |
|------|-------------|
| `MLPClassifier` | A fully reconstructed scikit-learn `MLPClassifier` ready to call `.predict()` and `.predict_proba()`. |

**Example**

```python
import sklearn
from sklearn_migrator.classification.mlp_clf import deserialize_mlp_clf

model = deserialize_mlp_clf(data, sklearn.__version__)
predictions = model.predict(X_test)
```

---

## Regression

### Linear Regression

```python
from sklearn_migrator.regression.linear_regression_reg import (
    serialize_linear_regression_reg,
    deserialize_linear_regression_reg,
)
```

---

#### `serialize_linear_regression_reg`

```python
serialize_linear_regression_reg(model: LinearRegression, version_in: str) -> dict
```

Converts a fitted `LinearRegression` into a JSON-compatible dictionary capturing coefficients, intercept, rank, and singular values.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `model` | `LinearRegression` | A fitted scikit-learn `LinearRegression` instance. |
| `version_in` | `str` | The sklearn version used to train the model (e.g. `'1.2.0'`). |

**Returns**

| Type | Description |
|------|-------------|
| `dict` | A JSON-serializable dictionary containing `coef_`, `intercept_`, `singular_`, `rank_`, hyperparameters, and version metadata. |

**Example**

```python
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn_migrator.regression.linear_regression_reg import serialize_linear_regression_reg

model = LinearRegression().fit(X_train, y_train)
data = serialize_linear_regression_reg(model, sklearn.__version__)
```

---

#### `deserialize_linear_regression_reg`

```python
deserialize_linear_regression_reg(data: dict, version_out: str) -> LinearRegression
```

Reconstructs a `LinearRegression` from a serialized dictionary, restoring all learned coefficients and metadata.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `data` | `dict` | Dictionary produced by `serialize_linear_regression_reg`. |
| `version_out` | `str` | The sklearn version of the target environment (e.g. `'1.7.0'`). |

**Returns**

| Type | Description |
|------|-------------|
| `LinearRegression` | A fully reconstructed scikit-learn `LinearRegression` ready to call `.predict()`. |

**Example**

```python
import sklearn
from sklearn_migrator.regression.linear_regression_reg import deserialize_linear_regression_reg

model = deserialize_linear_regression_reg(data, sklearn.__version__)
predictions = model.predict(X_test)
```

---

### Ridge Regression

```python
from sklearn_migrator.regression.ridge_reg import (
    serialize_ridge_reg,
    deserialize_ridge_reg,
)
```

---

#### `serialize_ridge_reg`

```python
serialize_ridge_reg(model: Ridge, version_in: str) -> dict
```

Converts a fitted `Ridge` regression model into a JSON-compatible dictionary capturing coefficients, intercept, and the regularization strength.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `model` | `Ridge` | A fitted scikit-learn `Ridge` instance. |
| `version_in` | `str` | The sklearn version used to train the model (e.g. `'1.2.0'`). |

**Returns**

| Type | Description |
|------|-------------|
| `dict` | A JSON-serializable dictionary containing `alpha`, `coef_`, `intercept_`, hyperparameters, and version metadata. |

**Example**

```python
import sklearn
from sklearn.linear_model import Ridge
from sklearn_migrator.regression.ridge_reg import serialize_ridge_reg

model = Ridge(alpha=1.0).fit(X_train, y_train)
data = serialize_ridge_reg(model, sklearn.__version__)
```

---

#### `deserialize_ridge_reg`

```python
deserialize_ridge_reg(data: dict, version_out: str) -> Ridge
```

Reconstructs a `Ridge` regression model from a serialized dictionary.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `data` | `dict` | Dictionary produced by `serialize_ridge_reg`. |
| `version_out` | `str` | The sklearn version of the target environment (e.g. `'1.7.0'`). |

**Returns**

| Type | Description |
|------|-------------|
| `Ridge` | A fully reconstructed scikit-learn `Ridge` ready to call `.predict()`. |

**Example**

```python
import sklearn
from sklearn_migrator.regression.ridge_reg import deserialize_ridge_reg

model = deserialize_ridge_reg(data, sklearn.__version__)
predictions = model.predict(X_test)
```

---

### Lasso Regression

```python
from sklearn_migrator.regression.lasso_reg import (
    serialize_lasso_reg,
    deserialize_lasso_reg,
)
```

---

#### `serialize_lasso_reg`

```python
serialize_lasso_reg(model: Lasso, version_in: str) -> dict
```

Converts a fitted `Lasso` regression model into a JSON-compatible dictionary capturing sparse coefficients, intercept, and the regularization strength.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `model` | `Lasso` | A fitted scikit-learn `Lasso` instance. |
| `version_in` | `str` | The sklearn version used to train the model (e.g. `'1.2.0'`). |

**Returns**

| Type | Description |
|------|-------------|
| `dict` | A JSON-serializable dictionary containing `alpha`, `coef_`, `intercept_`, hyperparameters, and version metadata. |

**Example**

```python
import sklearn
from sklearn.linear_model import Lasso
from sklearn_migrator.regression.lasso_reg import serialize_lasso_reg

model = Lasso(alpha=0.1).fit(X_train, y_train)
data = serialize_lasso_reg(model, sklearn.__version__)
```

---

#### `deserialize_lasso_reg`

```python
deserialize_lasso_reg(data: dict, version_out: str) -> Lasso
```

Reconstructs a `Lasso` regression model from a serialized dictionary.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `data` | `dict` | Dictionary produced by `serialize_lasso_reg`. |
| `version_out` | `str` | The sklearn version of the target environment (e.g. `'1.7.0'`). |

**Returns**

| Type | Description |
|------|-------------|
| `Lasso` | A fully reconstructed scikit-learn `Lasso` ready to call `.predict()`. |

**Example**

```python
import sklearn
from sklearn_migrator.regression.lasso_reg import deserialize_lasso_reg

model = deserialize_lasso_reg(data, sklearn.__version__)
predictions = model.predict(X_test)
```

---

### Decision Tree Regressor

```python
from sklearn_migrator.regression.decision_tree_reg import (
    serialize_decision_tree_reg,
    deserialize_decision_tree_reg,
)
```

---

#### `serialize_decision_tree_reg`

```python
serialize_decision_tree_reg(model: DecisionTreeRegressor, version_in: str) -> dict
```

Converts a fitted `DecisionTreeRegressor` into a JSON-compatible dictionary including the full internal node structure.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `model` | `DecisionTreeRegressor` | A fitted scikit-learn `DecisionTreeRegressor` instance. |
| `version_in` | `str` | The sklearn version used to train the model (e.g. `'1.2.0'`). |

**Returns**

| Type | Description |
|------|-------------|
| `dict` | A JSON-serializable dictionary containing the tree's node structure, leaf values, feature metadata, and hyperparameters. |

**Example**

```python
import sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn_migrator.regression.decision_tree_reg import serialize_decision_tree_reg

model = DecisionTreeRegressor(max_depth=5).fit(X_train, y_train)
data = serialize_decision_tree_reg(model, sklearn.__version__)
```

---

#### `deserialize_decision_tree_reg`

```python
deserialize_decision_tree_reg(data: dict, version_out: str) -> DecisionTreeRegressor
```

Reconstructs a `DecisionTreeRegressor` from a serialized dictionary, handling node-structure differences between sklearn versions automatically.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `data` | `dict` | Dictionary produced by `serialize_decision_tree_reg`. |
| `version_out` | `str` | The sklearn version of the target environment (e.g. `'1.7.0'`). |

**Returns**

| Type | Description |
|------|-------------|
| `DecisionTreeRegressor` | A fully reconstructed scikit-learn `DecisionTreeRegressor` ready to call `.predict()`. |

**Example**

```python
import sklearn
from sklearn_migrator.regression.decision_tree_reg import deserialize_decision_tree_reg

model = deserialize_decision_tree_reg(data, sklearn.__version__)
predictions = model.predict(X_test)
```

---

### Random Forest Regressor

```python
from sklearn_migrator.regression.random_forest_reg import (
    serialize_random_forest_reg,
    deserialize_random_forest_reg,
)
```

---

#### `serialize_random_forest_reg`

```python
serialize_random_forest_reg(model: RandomForestRegressor, version_in: str) -> dict
```

Converts a fitted `RandomForestRegressor` into a JSON-compatible dictionary by serializing each constituent decision tree individually.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `model` | `RandomForestRegressor` | A fitted scikit-learn `RandomForestRegressor` instance. |
| `version_in` | `str` | The sklearn version used to train the model (e.g. `'1.2.0'`). |

**Returns**

| Type | Description |
|------|-------------|
| `dict` | A JSON-serializable dictionary containing the serialized estimator list, ensemble hyperparameters, and version metadata. |

**Example**

```python
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn_migrator.regression.random_forest_reg import serialize_random_forest_reg

model = RandomForestRegressor(n_estimators=100).fit(X_train, y_train)
data = serialize_random_forest_reg(model, sklearn.__version__)
```

---

#### `deserialize_random_forest_reg`

```python
deserialize_random_forest_reg(data: dict, version_out: str) -> RandomForestRegressor
```

Reconstructs a `RandomForestRegressor` from a serialized dictionary, rebuilding all constituent trees for the target sklearn version.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `data` | `dict` | Dictionary produced by `serialize_random_forest_reg`. |
| `version_out` | `str` | The sklearn version of the target environment (e.g. `'1.7.0'`). |

**Returns**

| Type | Description |
|------|-------------|
| `RandomForestRegressor` | A fully reconstructed scikit-learn `RandomForestRegressor` ready to call `.predict()`. |

**Example**

```python
import sklearn
from sklearn_migrator.regression.random_forest_reg import deserialize_random_forest_reg

model = deserialize_random_forest_reg(data, sklearn.__version__)
predictions = model.predict(X_test)
```

---

### Gradient Boosting Regressor

```python
from sklearn_migrator.regression.gradient_boosting_reg import (
    serialize_gradient_boosting_reg,
    deserialize_gradient_boosting_reg,
)
```

---

#### `serialize_gradient_boosting_reg`

```python
serialize_gradient_boosting_reg(model: GradientBoostingRegressor, version_in: str) -> dict
```

Converts a fitted `GradientBoostingRegressor` into a JSON-compatible dictionary, including each boosting stage's regression tree and the initial estimator state.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `model` | `GradientBoostingRegressor` | A fitted scikit-learn `GradientBoostingRegressor` instance. |
| `version_in` | `str` | The sklearn version used to train the model (e.g. `'1.2.0'`). |

**Returns**

| Type | Description |
|------|-------------|
| `dict` | A JSON-serializable dictionary containing all boosting stage trees, the dummy regressor state, loss function name, training scores, and hyperparameters. |

**Example**

```python
import sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn_migrator.regression.gradient_boosting_reg import serialize_gradient_boosting_reg

model = GradientBoostingRegressor().fit(X_train, y_train)
data = serialize_gradient_boosting_reg(model, sklearn.__version__)
```

---

#### `deserialize_gradient_boosting_reg`

```python
deserialize_gradient_boosting_reg(data: dict, version_out: str) -> GradientBoostingRegressor
```

Reconstructs a `GradientBoostingRegressor` from a serialized dictionary, rewiring the loss function and initial estimator to match the target sklearn version.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `data` | `dict` | Dictionary produced by `serialize_gradient_boosting_reg`. |
| `version_out` | `str` | The sklearn version of the target environment (e.g. `'1.7.0'`). |

**Returns**

| Type | Description |
|------|-------------|
| `GradientBoostingRegressor` | A fully reconstructed scikit-learn `GradientBoostingRegressor` ready to call `.predict()`. |

**Example**

```python
import sklearn
from sklearn_migrator.regression.gradient_boosting_reg import deserialize_gradient_boosting_reg

model = deserialize_gradient_boosting_reg(data, sklearn.__version__)
predictions = model.predict(X_test)
```

---

### AdaBoost Regressor

```python
from sklearn_migrator.regression.adaboost_reg import (
    serialize_adaboost_reg,
    deserialize_adaboost_reg,
)
```

---

#### `serialize_adaboost_reg`

```python
serialize_adaboost_reg(model: AdaBoostRegressor, version_in: str) -> dict
```

Converts a fitted `AdaBoostRegressor` into a JSON-compatible dictionary, serializing each weak learner tree and its associated sample weights.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `model` | `AdaBoostRegressor` | A fitted scikit-learn `AdaBoostRegressor` instance. |
| `version_in` | `str` | The sklearn version used to train the model (e.g. `'1.2.0'`). |

**Returns**

| Type | Description |
|------|-------------|
| `dict` | A JSON-serializable dictionary containing the serialized estimator list, estimator weights, estimator errors, hyperparameters, and version metadata. |

**Example**

```python
import sklearn
from sklearn.ensemble import AdaBoostRegressor
from sklearn_migrator.regression.adaboost_reg import serialize_adaboost_reg

model = AdaBoostRegressor(n_estimators=50).fit(X_train, y_train)
data = serialize_adaboost_reg(model, sklearn.__version__)
```

---

#### `deserialize_adaboost_reg`

```python
deserialize_adaboost_reg(data: dict, version_out: str) -> AdaBoostRegressor
```

Reconstructs an `AdaBoostRegressor` from a serialized dictionary, automatically handling the `base_estimator` / `estimator` rename that occurred between sklearn versions.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `data` | `dict` | Dictionary produced by `serialize_adaboost_reg`. |
| `version_out` | `str` | The sklearn version of the target environment (e.g. `'1.7.0'`). |

**Returns**

| Type | Description |
|------|-------------|
| `AdaBoostRegressor` | A fully reconstructed scikit-learn `AdaBoostRegressor` ready to call `.predict()`. |

**Example**

```python
import sklearn
from sklearn_migrator.regression.adaboost_reg import deserialize_adaboost_reg

model = deserialize_adaboost_reg(data, sklearn.__version__)
predictions = model.predict(X_test)
```

---

### MLP Regressor

```python
from sklearn_migrator.regression.mlp_reg import (
    serialize_mlp_reg,
    deserialize_mlp_reg,
)
```

---

#### `serialize_mlp_reg`

```python
serialize_mlp_reg(model: MLPRegressor, version_in: str) -> dict
```

Converts a fitted `MLPRegressor` into a JSON-compatible dictionary capturing all layer weights, biases, and training metadata.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `model` | `MLPRegressor` | A fitted scikit-learn `MLPRegressor` instance. |
| `version_in` | `str` | The sklearn version used to train the model (e.g. `'1.2.0'`). |

**Returns**

| Type | Description |
|------|-------------|
| `dict` | A JSON-serializable dictionary containing all weight matrices (`coefs_`), bias vectors (`intercepts_`), training loss, layer counts, output activation, and hyperparameters. |

**Example**

```python
import sklearn
from sklearn.neural_network import MLPRegressor
from sklearn_migrator.regression.mlp_reg import serialize_mlp_reg

model = MLPRegressor(hidden_layer_sizes=(64, 32)).fit(X_train, y_train)
data = serialize_mlp_reg(model, sklearn.__version__)
```

---

#### `deserialize_mlp_reg`

```python
deserialize_mlp_reg(data: dict, version_out: str) -> MLPRegressor
```

Reconstructs a `MLPRegressor` from a serialized dictionary, restoring all weights, biases, and network architecture.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `data` | `dict` | Dictionary produced by `serialize_mlp_reg`. |
| `version_out` | `str` | The sklearn version of the target environment (e.g. `'1.7.0'`). |

**Returns**

| Type | Description |
|------|-------------|
| `MLPRegressor` | A fully reconstructed scikit-learn `MLPRegressor` ready to call `.predict()`. |

**Example**

```python
import sklearn
from sklearn_migrator.regression.mlp_reg import deserialize_mlp_reg

model = deserialize_mlp_reg(data, sklearn.__version__)
predictions = model.predict(X_test)
```

---

### Support Vector Regressor (SVR)

```python
from sklearn_migrator.regression.svm_reg import (
    serialize_svr,
    deserialize_svr,
)
```

> **Important:** `deserialize_svr` returns a `Migrated_SVR` object, not a native `sklearn.svm.SVR`. `Migrated_SVR` is a pure-NumPy reimplementation that avoids sklearn's compiled C extensions, making it stable across all supported versions. It exposes `.predict()` and supports the `linear`, `rbf`, `poly`, and `sigmoid` kernels.

---

#### `serialize_svr`

```python
serialize_svr(model: SVR, version_in: str) -> dict
```

Converts a fitted `SVR` into a JSON-compatible dictionary capturing support vectors, dual coefficients, intercept, and kernel parameters.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `model` | `SVR` | A fitted scikit-learn `SVR` instance. |
| `version_in` | `str` | The sklearn version used to train the model (e.g. `'1.2.0'`). |

**Returns**

| Type | Description |
|------|-------------|
| `dict` | A JSON-serializable dictionary containing support vectors, dual coefficients, internal intercept, gamma, kernel parameters, and version metadata. |

**Example**

```python
import sklearn
from sklearn.svm import SVR
from sklearn_migrator.regression.svm_reg import serialize_svr

model = SVR(kernel='rbf', C=1.0).fit(X_train, y_train)
data = serialize_svr(model, sklearn.__version__)
```

---

#### `deserialize_svr`

```python
deserialize_svr(data: dict, version_out: str) -> Migrated_SVR
```

Reconstructs a `Migrated_SVR` from a serialized dictionary, providing a version-independent prediction interface.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `data` | `dict` | Dictionary produced by `serialize_svr`. |
| `version_out` | `str` | The sklearn version of the target environment (e.g. `'1.7.0'`). |

**Returns**

| Type | Description |
|------|-------------|
| `Migrated_SVR` | A pure-NumPy regressor with a `.predict()` method compatible with any numpy array input. |

**Example**

```python
import sklearn
from sklearn_migrator.regression.svm_reg import deserialize_svr

model = deserialize_svr(data, sklearn.__version__)
predictions = model.predict(X_test)
```

---

### K-Nearest Neighbors Regressor

```python
from sklearn_migrator.regression.knn_reg import (
    serialize_knn_reg,
    deserialize_knn_reg,
)
```

---

#### `serialize_knn_reg`

```python
serialize_knn_reg(model: KNeighborsRegressor, version_in: str) -> dict
```

Converts a fitted `KNeighborsRegressor` into a JSON-compatible dictionary by capturing the full training data and hyperparameters.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `model` | `KNeighborsRegressor` | A fitted scikit-learn `KNeighborsRegressor` instance. |
| `version_in` | `str` | The sklearn version used to train the model (e.g. `'1.2.0'`). |

**Returns**

| Type | Description |
|------|-------------|
| `dict` | A JSON-serializable dictionary containing the training data matrix, target values, and all hyperparameters. |

> **Note:** Because KNN is lazy, the full training set is stored. Dictionary size scales with the number of training samples.

**Example**

```python
import sklearn
from sklearn.neighbors import KNeighborsRegressor
from sklearn_migrator.regression.knn_reg import serialize_knn_reg

model = KNeighborsRegressor(n_neighbors=5).fit(X_train, y_train)
data = serialize_knn_reg(model, sklearn.__version__)
```

---

#### `deserialize_knn_reg`

```python
deserialize_knn_reg(data: dict, version_out: str) -> KNeighborsRegressor
```

Reconstructs a `KNeighborsRegressor` by re-fitting it on the stored training data using the original hyperparameters.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `data` | `dict` | Dictionary produced by `serialize_knn_reg`. |
| `version_out` | `str` | The sklearn version of the target environment (e.g. `'1.7.0'`). |

**Returns**

| Type | Description |
|------|-------------|
| `KNeighborsRegressor` | A fully reconstructed scikit-learn `KNeighborsRegressor` ready to call `.predict()`. |

**Example**

```python
import sklearn
from sklearn_migrator.regression.knn_reg import deserialize_knn_reg

model = deserialize_knn_reg(data, sklearn.__version__)
predictions = model.predict(X_test)
```

---

## Clustering

### KMeans

```python
from sklearn_migrator.clustering.k_means import (
    serialize_k_means,
    deserialize_k_means,
)
```

---

#### `serialize_k_means`

```python
serialize_k_means(model: KMeans, version_in: str) -> dict
```

Converts a fitted `KMeans` into a JSON-compatible dictionary capturing cluster centers, labels, inertia, and all hyperparameters.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `model` | `KMeans` | A fitted scikit-learn `KMeans` instance. |
| `version_in` | `str` | The sklearn version used to train the model (e.g. `'1.2.0'`). |

**Returns**

| Type | Description |
|------|-------------|
| `dict` | A JSON-serializable dictionary containing `cluster_centers_`, `labels_`, `inertia_`, `n_iter_`, hyperparameters, and version metadata. Deprecated parameters (`n_jobs`, `precompute_distances`) are stripped automatically. |

**Example**

```python
import sklearn
from sklearn.cluster import KMeans
from sklearn_migrator.clustering.k_means import serialize_k_means

model = KMeans(n_clusters=3, random_state=42).fit(X_train)
data = serialize_k_means(model, sklearn.__version__)
```

---

#### `deserialize_k_means`

```python
deserialize_k_means(data: dict, version_out: str) -> KMeans
```

Reconstructs a `KMeans` from a serialized dictionary, restoring cluster centers, labels, and all internal state.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `data` | `dict` | Dictionary produced by `serialize_k_means`. |
| `version_out` | `str` | The sklearn version of the target environment (e.g. `'1.7.0'`). |

**Returns**

| Type | Description |
|------|-------------|
| `KMeans` | A fully reconstructed scikit-learn `KMeans` ready to call `.predict()` and `.transform()`. |

**Example**

```python
import sklearn
from sklearn_migrator.clustering.k_means import deserialize_k_means

model = deserialize_k_means(data, sklearn.__version__)
cluster_labels = model.predict(X_test)
```

---

### MiniBatchKMeans

```python
from sklearn_migrator.clustering.mini_batch_k_means import (
    serialize_mini_batch_kmeans,
    deserialize_mini_batch_kmeans,
)
```

---

#### `serialize_mini_batch_kmeans`

```python
serialize_mini_batch_kmeans(model: MiniBatchKMeans, version_in: str) -> dict
```

Converts a fitted `MiniBatchKMeans` into a JSON-compatible dictionary capturing cluster centers, training metadata, and hyperparameters.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `model` | `MiniBatchKMeans` | A fitted scikit-learn `MiniBatchKMeans` instance. |
| `version_in` | `str` | The sklearn version used to train the model (e.g. `'1.2.0'`). |

**Returns**

| Type | Description |
|------|-------------|
| `dict` | A JSON-serializable dictionary containing `cluster_centers_`, `labels_`, `inertia_`, `n_iter_`, hyperparameters (with version-deprecated keys stripped), and version metadata. |

**Example**

```python
import sklearn
from sklearn.cluster import MiniBatchKMeans
from sklearn_migrator.clustering.mini_batch_k_means import serialize_mini_batch_kmeans

model = MiniBatchKMeans(n_clusters=3, batch_size=256).fit(X_train)
data = serialize_mini_batch_kmeans(model, sklearn.__version__)
```

---

#### `deserialize_mini_batch_kmeans`

```python
deserialize_mini_batch_kmeans(data: dict, version_out: str) -> MiniBatchKMeans
```

Reconstructs a `MiniBatchKMeans` from a serialized dictionary, restoring cluster centers and all internal state.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `data` | `dict` | Dictionary produced by `serialize_mini_batch_kmeans`. |
| `version_out` | `str` | The sklearn version of the target environment (e.g. `'1.7.0'`). |

**Returns**

| Type | Description |
|------|-------------|
| `MiniBatchKMeans` | A fully reconstructed scikit-learn `MiniBatchKMeans` ready to call `.predict()` and `.transform()`. |

**Example**

```python
import sklearn
from sklearn_migrator.clustering.mini_batch_k_means import deserialize_mini_batch_kmeans

model = deserialize_mini_batch_kmeans(data, sklearn.__version__)
cluster_labels = model.predict(X_test)
```

---

### Agglomerative Clustering

```python
from sklearn_migrator.clustering.agglomerative import (
    serialize_agglomerative,
    deserialize_agglomerative,
)
```

---

#### `serialize_agglomerative`

```python
serialize_agglomerative(model: AgglomerativeClustering, version_in: str) -> dict
```

Converts a fitted `AgglomerativeClustering` into a JSON-compatible dictionary capturing the dendrogram structure, labels, and hyperparameters.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `model` | `AgglomerativeClustering` | A fitted scikit-learn `AgglomerativeClustering` instance. |
| `version_in` | `str` | The sklearn version used to train the model (e.g. `'1.2.0'`). |

**Returns**

| Type | Description |
|------|-------------|
| `dict` | A JSON-serializable dictionary containing `labels_`, `children_` (merge tree), `distances_`, `n_connected_components_`, hyperparameters (with the `affinity` to `metric` rename handled), and version metadata. |

**Example**

```python
import sklearn
from sklearn.cluster import AgglomerativeClustering
from sklearn_migrator.clustering.agglomerative import serialize_agglomerative

model = AgglomerativeClustering(n_clusters=3).fit(X_train)
data = serialize_agglomerative(model, sklearn.__version__)
```

---

#### `deserialize_agglomerative`

```python
deserialize_agglomerative(data: dict, version_out: str) -> AgglomerativeClustering
```

Reconstructs an `AgglomerativeClustering` from a serialized dictionary, handling the `affinity` / `metric` parameter rename across sklearn versions automatically.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `data` | `dict` | Dictionary produced by `serialize_agglomerative`. |
| `version_out` | `str` | The sklearn version of the target environment (e.g. `'1.7.0'`). |

**Returns**

| Type | Description |
|------|-------------|
| `AgglomerativeClustering` | A fully reconstructed scikit-learn `AgglomerativeClustering` with all labels and hierarchy information restored. |

**Example**

```python
import sklearn
from sklearn_migrator.clustering.agglomerative import deserialize_agglomerative

model = deserialize_agglomerative(data, sklearn.__version__)
print(model.labels_)
```

---

## Dimensionality Reduction

### PCA

```python
from sklearn_migrator.dimension.pca import (
    serialize_pca,
    deserialize_pca,
)
```

---

#### `serialize_pca`

```python
serialize_pca(model: PCA, version_in: str) -> dict
```

Converts a fitted `PCA` into a JSON-compatible dictionary capturing principal components, explained variance, mean, and singular values.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `model` | `PCA` | A fitted scikit-learn `PCA` instance. |
| `version_in` | `str` | The sklearn version used to train the model (e.g. `'1.2.0'`). |

**Returns**

| Type | Description |
|------|-------------|
| `dict` | A JSON-serializable dictionary containing `components_`, `explained_variance_`, `explained_variance_ratio_`, `singular_values_`, `mean_`, `noise_variance_`, hyperparameters (with version-specific keys stripped), and version metadata. |

**Example**

```python
import sklearn
from sklearn.decomposition import PCA
from sklearn_migrator.dimension.pca import serialize_pca

model = PCA(n_components=10).fit(X_train)
data = serialize_pca(model, sklearn.__version__)

import json
with open("pca.json", "w") as f:
    json.dump(data, f)
```

---

#### `deserialize_pca`

```python
deserialize_pca(data: dict, version_out: str) -> PCA
```

Reconstructs a `PCA` from a serialized dictionary, restoring all principal components and variance statistics for the target sklearn version.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `data` | `dict` | Dictionary produced by `serialize_pca`. |
| `version_out` | `str` | The sklearn version of the target environment (e.g. `'1.7.0'`). |

**Returns**

| Type | Description |
|------|-------------|
| `PCA` | A fully reconstructed scikit-learn `PCA` ready to call `.transform()` and `.inverse_transform()`. |

**Example**

```python
import json
import sklearn
from sklearn_migrator.dimension.pca import deserialize_pca

with open("pca.json") as f:
    data = json.load(f)

model = deserialize_pca(data, sklearn.__version__)
X_reduced = model.transform(X_test)
```

---

## Common Patterns

### End-to-end migration

The standard workflow is identical for every model: serialize on the source machine, save to disk, load on the target machine, deserialize.

```python
# --- Source environment (e.g. sklearn 0.24.1) ---
import json
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn_migrator.classification.random_forest_clf import serialize_random_forest_clf

model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
data = serialize_random_forest_clf(model, sklearn.__version__)

with open("model.json", "w") as f:
    json.dump(data, f)

# --- Target environment (e.g. sklearn 1.7.0) ---
import json
import sklearn
from sklearn_migrator.classification.random_forest_clf import deserialize_random_forest_clf

with open("model.json") as f:
    data = json.load(f)

model = deserialize_random_forest_clf(data, sklearn.__version__)
predictions = model.predict(X_test)
```

### Warnings about incompatible fields

When a field exists in one sklearn version but not another, the library silently skips it.
If a field is present but cannot be set, a `UserWarning` is issued with the field name and the error reason. These warnings are informational — the reconstructed model will still work correctly for inference.
