# sklearn-migrator 🧪

**A Python library to serialize and migrate scikit-learn models across incompatible versions.**

[![PyPI version](https://img.shields.io/pypi/v/sklearn-migrator.svg)](https://pypi.org/project/sklearn-migrator/)
![Python versions](https://img.shields.io/pypi/pyversions/sklearn-migrator.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/anvaldes/sklearn-migrator/actions/workflows/ci.yml/badge.svg)](https://github.com/anvaldes/sklearn-migrator/actions)
[![codecov](https://codecov.io/gh/anvaldes/sklearn-migrator/branch/main/graph/badge.svg)](https://codecov.io/gh/anvaldes/sklearn-migrator)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17917931.svg)](https://doi.org/10.5281/zenodo.17917931)


<p align="center">
  <img src="images/Logo_Lateral.png" alt="sklearn-migrator" width="200"/>
</p>

---

# 🚀 Motivation

Machine learning teams frequently store trained scikit-learn models using `pickle` or `joblib`.  
However:

### ❌ These serialized models **break** when scikit-learn versions change  
- Internal attributes change  
- APIs evolve (e.g., `affinity → metric`)  
- Tree and boosting internals get reorganized  
- New default parameters appear  

### ❌ This creates real problems:
- Production services fail after dependency upgrades  
- Research becomes non-reproducible  
- Long-term model governance becomes impossible  
- Models can't be migrated or audited reliably  

---


# ✅ What `sklearn-migrator` provides

### ✔ Serialize any supported model into a **JSON-compatible dictionary**  
### ✔ Deserialize and reconstruct the model **in a different scikit-learn version**  
### ✔ Remove dependency on pickle/joblib for long-term storage  
### ✔ Enable reproducible ML pipelines across environments  

This library has been validated across **1,024 version migration pairs** (from → to), covering:

0.21.3 → 1.7.2

---

# 💡 Supported Models (21 models)

`sklearn-migrator` supports **21 core models** across classification, regression, clustering, and dimensionality reduction.

## 📘 Classification

| Model                        | Supported |
|------------------------------|-----------|
| DecisionTreeClassifier       | ✅ |
| RandomForestClassifier       | ✅ |
| GradientBoostingClassifier   | ✅ |
| LogisticRegression           | ✅ |
| KNeighborsClassifier         | ✅ |
| SVC (Support Vector Classifier) | ✅ |
| MLPClassifier                | ✅ |

---

## 📗 Regression

| Model                        | Supported |
|------------------------------|-----------|
| DecisionTreeRegressor        | ✅ |
| RandomForestRegressor        | ✅ |
| GradientBoostingRegressor    | ✅ |
| LinearRegression             | ✅ |
| Ridge                        | ✅ |
| Lasso                        | ✅ |
| KNeighborsRegressor          | ✅ |
| SVR (Support Vector Regressor) | ✅ |
| AdaBoostRegressor            | ✅ |
| MLPRegressor                 | ✅ |

---

## 📙 Clustering

| Model                | Supported |
|----------------------|-----------|
| KMeans               | ✅ |
| MiniBatchKMeans      | ✅ |
| Agglomerative        | ✅ |


---

## 📘 Dimensionality Reduction

| Model | Supported |
|-------|-----------|
| PCA   | ✅ |

---

## 🔢 Version Compatibility Matrix

The library supports model migrations across the full matrix:

- **32 versions**  
- **1,024 migration pairs**  
- Fully tested using automated environments via CI/CD on every push

```python
versions = [
    '0.21.3', '0.22.0', '0.22.1', '0.23.0', '0.23.1', '0.23.2',
    '0.24.0', '0.24.1', '0.24.2', '1.0.0', '1.0.1', '1.0.2',
    '1.1.0', '1.1.1', '1.1.2', '1.1.3', '1.2.0', '1.2.1', '1.2.2',
    '1.3.0', '1.3.1', '1.3.2', '1.4.0', '1.4.2', '1.5.0', '1.5.1',
    '1.5.2', '1.6.0', '1.6.1', '1.7.0', '1.7.1', '1.7.2'
]
```

| From \ To | 0.21.3 | 0.22.0 | ... | 1.7.2 |
| --------- | ------ | ------ | --- | ----- |
| 0.21.3    | ✅      | ✅      | ... | ✅     |
| 0.22.0    | ✅      | ✅      | ... | ✅     |
| ...       | ...    | ...    | ... | ...   |
| 1.7.2     | ✅      | ✅      | ... | ✅     |

### 📊 Validation Metric

Each migration pair `(version_in, version_out)` is validated using:

$$\max |y_{\text{in}} - y_{\text{out}}| < 10^{-2}$$

Where:
- $y_{\text{in}}$: predictions from the model in the **source** version
- $y_{\text{out}}$: predictions from the migrated model in the **target** version

The worst case across all 1,024 pairs is obtained via:

```python
df_performance.abs().max().max()  # global worst case (32x32 matrix)
```

> ⚠️ All 1,024 combinations and 21 models are automatically tested on every push via CI/CD, using isolated Docker environments for each sklearn version. Each model is validated under a representative parameter configuration; exhaustive combinatorial testing of all parameter combinations is outside the current scope.

---

## 📂 Installation

```bash
pip install sklearn-migrator
```

---

## 💥 Use Cases

* **Long-term model storage**: Store models in a future-proof format across teams and systems.
* **Production model migration**: Move models safely across major `scikit-learn` upgrades.
* **Auditing and inspection**: Read serialized models as JSON, inspect structure, hyperparameters, and internals.
* **Cross-platform inference**: Serialize in Python, serve elsewhere (e.g., microservices).

---

## 1. Using two python environments

You can serialize the model from an environment with a scikit-learn version (for example `1.5.0`) and then deserialize the model from another environment with a different version (for example `1.7.0`).

The deserialized model has the version of the environment where you deserialized it. In this case `1.7.0`.

It is important to understand what version of scikit-learn you want to migrate from, and what version you want to migrate to, in order to create the appropriate environments for serialization and deserialization.

### a. Serialize the model

```python
import json
import sklearn
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn_migrator.regression.random_forest_reg import serialize_random_forest_reg

version_sklearn_in = sklearn.__version__

model = RandomForestRegressor()
model.fit(X_train, y_train)

# If you want to compare output from this model and the new model with its new version
y_pred = pd.DataFrame(model.predict(X_test))
y_pred.to_csv('y_pred.csv', index=False)

all_data = serialize_random_forest_reg(model, version_sklearn_in)

# Save it

def convert(o):
    if isinstance(o, (np.integer, np.int64)):
        return int(o)
    elif isinstance(o, (np.floating, np.float64)):
        return float(o)
    elif isinstance(o, np.ndarray):
        return o.tolist()
    else:
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

with open("/input/model.json", "w") as f:
    json.dump(all_data, f, default=convert)
```

### b. Deserialize the model

```python
import json
import sklearn
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn_migrator.regression.random_forest_reg import deserialize_random_forest_reg

version_sklearn_out = sklearn.__version__

with open("/input/model.json", "r") as f:
    all_data = json.load(f)

new_model = deserialize_random_forest_reg(all_data, version_sklearn_out)

# Now you have your model in this new version

# If you want to compare the outputs
y_pred_new = pd.DataFrame(new_model.predict(X_test))
y_pred_new.to_csv('y_pred_new.csv', index=False)

# Compare "y_pred.csv" with "y_pred_new.csv"
```

## 2. Docker: Step by Step

You have a Random Forest Classifier saved in a `.pkl` format and it is called `model.pkl`. The version of this model is `1.5.0`.

i. Create in your Desktop the next folder:

```bash
/test_github
```

And copy your `model.pkl` in this folder.

ii. The Dockerfiles and requirements for all supported input versions are available in the `integration/environments/input/` directory of this repository. Copy the files for your input version (e.g., `1.5.0`):

```bash
/test_github/input/1.5.0/Dockerfile_input
/test_github/input/1.5.0/requirements_input.txt
```

iii. The Dockerfiles and requirements for all supported output versions are available in the `integration/environments/output/` directory of this repository. Copy the files for your output version (e.g., `1.7.0`):

```bash
/test_github/output/1.7.0/Dockerfile_output
/test_github/output/1.7.0/requirements_output.txt
```

iv. Now you create your `input.py`:

```python
import json
import joblib
import sklearn
import numpy as np
import pandas as pd
from joblib import load

from sklearn.ensemble import RandomForestClassifier

from sklearn_migrator.classification.random_forest_clf import serialize_random_forest_clf

version_sklearn_in = sklearn.__version__

model = load('model.pkl')

all_data = serialize_random_forest_clf(model, version_sklearn_in)

def convert(o):
    if isinstance(o, (np.integer, np.int64)):
        return int(o)
    elif isinstance(o, (np.floating, np.float64)):
        return float(o)
    elif isinstance(o, np.ndarray):
        return o.tolist()
    else:
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

with open("input_model/all_data.json", "w") as f:
    json.dump(all_data, f, default=convert)

fake_row = np.array([[0.5, -1.2, 0.3, 1.1, -0.7, 0.9, 0.0, -0.3, 1.5, 0.2]])

y_pred = pd.DataFrame(model.predict_proba(fake_row))
y_pred.to_csv('input_model/y_pred.csv', index=False)
```

v. Now you create your `output.py`:

```python
import json
import joblib
import sklearn
import numpy as np
import pandas as pd
from joblib import load

from sklearn.ensemble import RandomForestClassifier

from sklearn_migrator.classification.random_forest_clf import deserialize_random_forest_clf

version_sklearn_out = sklearn.__version__

with open("input_model/all_data.json", "r") as f:
    all_data = json.load(f)

new_model = deserialize_random_forest_clf(all_data, version_sklearn_out)

joblib.dump(new_model, 'output_model/new_model.pkl')

fake_row = np.array([[0.5, -1.2, 0.3, 1.1, -0.7, 0.9, 0.0, -0.3, 1.5, 0.2]])

y_pred_new = pd.DataFrame(new_model.predict_proba(fake_row))
y_pred_new.to_csv('output_model/y_pred_new.csv', index=False)
```

vi. Now you copy all the files:

```bash
cp input/1.5.0/* output/1.7.0/* .
```

vii. Now you create two folders: `input_model/` and `output_model/`.

viii. Execute the next commands in your terminal (you should be in the root of `test_github/` folder):

```bash
docker build -f Dockerfile_input -t image_input_1.5.0 .
docker build -f Dockerfile_output -t image_output_1.7.0 .

docker run --rm \
  -v "$(pwd)/input_model:/app/input_model" \
  -v "$(pwd)/model.pkl:/app/model.pkl" \
  image_input_1.5.0

docker run --rm \
  -v "$(pwd)/input_model:/app/input_model" \
  -v "$(pwd)/output_model:/app/output_model" \
  image_output_1.7.0
```

ix. Finally you can find your migrated model in the folder `/output_model` and its name is `new_model.pkl`. This model is a scikit-learn model of version `1.7.0`.

---

## 🧾 Function Signatures & Parameters

Each model in `sklearn-migrator` provides a pair of functions:

- `serialize_<model_name>(model, version_in)`  
  Converts a trained scikit-learn model into a portable, version-agnostic dictionary.

- `deserialize_<model_name>(data, version_out)`  
  Reconstructs a scikit-learn model in a specific target version using the serialized dictionary.

### 🧠 Classification Models

```python
from sklearn_migrator.classification.decision_tree_clf import (
    serialize_decision_tree_clf,
    deserialize_decision_tree_clf,
)

from sklearn_migrator.classification.gradient_boosting_clf import (
    serialize_gradient_boosting_clf,
    deserialize_gradient_boosting_clf,
)

from sklearn_migrator.classification.knn_clf import (
    serialize_knn_clf,
    deserialize_knn_clf,
)

from sklearn_migrator.classification.logistic_regression_clf import (
    serialize_logistic_regression_clf,
    deserialize_logistic_regression_clf,
)

from sklearn_migrator.classification.mlp_clf import (
    serialize_mlp_clf,
    deserialize_mlp_clf,
)

from sklearn_migrator.classification.random_forest_clf import (
    serialize_random_forest_clf,
    deserialize_random_forest_clf,
)

from sklearn_migrator.classification.svm_clf import (
    serialize_svc,
    deserialize_svc,
)
```

### 🧮 Regression Models

```python
from sklearn_migrator.regression.adaboost_reg import (
    serialize_adaboost_reg,
    deserialize_adaboost_reg,
)

from sklearn_migrator.regression.decision_tree_reg import (
    serialize_decision_tree_reg,
    deserialize_decision_tree_reg,
)

from sklearn_migrator.regression.gradient_boosting_reg import (
    serialize_gradient_boosting_reg,
    deserialize_gradient_boosting_reg,
)

from sklearn_migrator.regression.knn_reg import (
    serialize_knn_reg,
    deserialize_knn_reg,
)

from sklearn_migrator.regression.lasso_reg import (
    serialize_lasso_reg,
    deserialize_lasso_reg,
)

from sklearn_migrator.regression.linear_regression_reg import (
    serialize_linear_regression_reg,
    deserialize_linear_regression_reg,
)

from sklearn_migrator.regression.mlp_reg import (
    serialize_mlp_reg,
    deserialize_mlp_reg,
)

from sklearn_migrator.regression.random_forest_reg import (
    serialize_random_forest_reg,
    deserialize_random_forest_reg,
)

from sklearn_migrator.regression.ridge_reg import (
    serialize_ridge_reg,
    deserialize_ridge_reg,
)

from sklearn_migrator.regression.svm_reg import (
    serialize_svr,
    deserialize_svr,
)
```

### 🧩 Clustering Models

```python
from sklearn_migrator.clustering.agglomerative import (
    serialize_agglomerative,
    deserialize_agglomerative,
)

from sklearn_migrator.clustering.k_means import (
    serialize_k_means,
    deserialize_k_means,
)

from sklearn_migrator.clustering.mini_batch_k_means import (
    serialize_mini_batch_kmeans,
    deserialize_mini_batch_kmeans,
)
```

### 📉 Dimensionality Reduction

```python
from sklearn_migrator.dimension.pca import (
    serialize_pca,
    deserialize_pca,
)
```

---

## 🔧 Development

### Run tests locally

```bash
pytest tests/
```

Integration tests run automatically on every push via CI/CD.

---

## 🤝 Contributing

1. Fork the repository
2. Create a new branch `feature/my-feature`
3. Open a pull request
4. Please ensure your pull request is tested for all combinations of functions; otherwise, it may be rejected.

We welcome bug reports, suggestions, and contributions of new models.

---

## 📄 License

MIT License — see [`LICENSE`](LICENSE) for details.

---

## 🔍 Author

**Alberto Valdés**

ML/AI Engineer | MLOps Engineer | Open Source Contributor

GitHub: [@anvaldes](https://github.com/anvaldes)

---