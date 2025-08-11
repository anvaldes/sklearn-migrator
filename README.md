# sklearn-migrator 🧪

**A Python library to serialize and migrate scikit-learn models across incompatible versions.**

[![PyPI version](https://img.shields.io/pypi/v/sklearn-migrator.svg)](https://pypi.org/project/sklearn-migrator/)
![Python versions](https://img.shields.io/pypi/pyversions/sklearn-migrator.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/anvaldes/sklearn-migrator/actions/workflows/ci.yml/badge.svg)](https://github.com/anvaldes/sklearn-migrator/actions)
[![codecov](https://codecov.io/gh/anvaldes/sklearn-migrator/branch/main/graph/badge.svg)](https://codecov.io/gh/anvaldes/sklearn-migrator)
[![DOI](https://zenodo.org/badge/DOI/10.xxxxx/zenodo.xxxxx.svg)](https://doi.org/10.xxxxx/zenodo.xxxxx)


<p align="center">
  <img src="images/Logo_Lateral.png" alt="sklearn-migrator" width="200"/>
</p>

---

## 🚀 Motivation

Serialized models using `joblib` or `pickle` are often incompatible between versions of `scikit-learn`, making it difficult to:

* Deploy models in production after library upgrades
* Migrate models across environments
* Share models across teams with different dependencies

**`sklearn-migrator`** allows you to:

* ✅ Serialize models into portable Python dictionaries (JSON-compatible)
* ✅ Migrate models across `scikit-learn` versions
* ✅ Inspect model structure without using `pickle`
* ✅ Ensure reproducibility in long-term ML projects

---

## 💡 Supported Models

### Classification Models

| Model                      | Supported |
| -------------------------- | --------- |
| DecisionTreeClassifier     | ✅         |
| RandomForestClassifier     | ✅         |
| GradientBoostingClassifier | ✅         |
| LogisticRegression         | ✅         |

### Regression Models

| Model                     | Supported |
| ------------------------- | --------- |
| DecisionTreeRegressor     | ✅         |
| RandomForestRegressor     | ✅         |
| GradientBoostingRegressor | ✅         |
| LinearRegression          | ✅         |

We’re actively expanding support for more models. If you’d like to contribute, we’d love your help! Feel free to open a pull request or suggest new features.

---

## 🔢 Version Compatibility Matrix

This library supports migration across the following `scikit-learn` versions:

```python
versions = [
    '0.21.3', '0.22.0', '0.22.1', '0.23.0', '0.23.1', '0.23.2',
    '0.24.0', '0.24.1', '0.24.2', '1.0.0', '1.0.1', '1.0.2',
    '1.1.0', '1.1.1', '1.1.2', '1.1.3', '1.2.0', '1.2.1', '1.2.2',
    '1.3.0', '1.3.1', '1.3.2', '1.4.0', '1.4.2', '1.5.0', '1.5.1',
    '1.5.2', '1.6.0', '1.6.1', '1.7.0'
]
```

There are 900 migration pairs (from-version → to-version).

| From \ To | 0.21.3 | 0.22.0 | ... | 1.7.0 |
| --------- | ------ | ------ | --- | ----- |
| 0.21.3    | ✅      | ✅      | ... | ✅     |
| 0.22.0    | ✅      | ✅      | ... | ✅     |
| ...       | ...    | ...    | ... | ...   |
| 1.7.0     | ✅      | ✅      | ... | ✅     |

> ⚠️ All 900 combinations were tested and validated using unit tests across real environments.

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

You can serialized the model from a environment with a scikit-learn version (for example `1.5.0`) and then you can deserialized the model from another environment with a other version (for exmaple `1.7.0`).

The deserialized model has the version of the environment where you deserialized it. In this case `1.7.0`.

As you can see is very important understand what is the version of scikit learn from you want to migrate to create and environment with this version to deserialized the model. On the other side you have to understand to what version you want to migrate to again create and environment with this version of scikit-learn.

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

# If you want to compare output from this model and the new model with his new version
y_pred = pd.DataFrame(model.predict(X_test))
y_pred.to_csv('y_pred.csv', index = False)

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

### b. Desarialize the model

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

new_model_reg_rfr = deserialize_random_forest_reg(all_data, version_sklearn_out)

# Now you have your model in this new version

# If you want to compare the outputs
y_pred_new = pd.DataFrame(new_model.predict(X_test))
y_pred_new.to_csv('y_pred_new.csv', index = False)

# Of course you compare "y_pred.csv" with "y_pred_new.csv"
```

## 2. Docker: Step by Step

You have a Random Forest Classifier saved in a `.pkl` format and it is called `model.pkl`. The version of this model is `1.5.0`.

i. Create in your Desktop the next folder:

```bash
/test_github
```

And copy your `model.pkl` in this folder.

ii. Go to the repo: https://github.com/anvaldes/environments_scikit_learn and find out for the Dockerfile and requirements.txt for the corresponding input scikit-learn version.

You will have this:

```bash
/test_github/input/1.5.0/Dockerfile_input
/test_github/input/1.5.0/requirements_input.txt
```

iii. Go to the repo: https://github.com/anvaldes/environments_scikit_learn and find out for the Dockerfile and requirements.txt for the output corresponding scikit-learn version.

You will have this:

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
y_pred.to_csv('input_model/y_pred.csv', index = False)
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
y_pred_new.to_csv('output_model/y_pred_new.csv', index = False)
```

vi. Now you copy all the files:

```bash
cp input/1.5.0/* output/1.7.0/* .
```

vii. Now you create two folders: `input_model/` and `output_model/`.

viii. Execute the next commands in your terminal (you should be in the root of `test_github/` folder)

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

### 🧮 Regression Models

```python
from sklearn_migrator.regression.decision_tree_reg import (
    serialize_decision_tree_reg,
    deserialize_decision_tree_reg
)

from sklearn_migrator.regression.linear_regression_reg import (
    serialize_linear_regression_reg,
    deserialize_linear_regression_reg
)

from sklearn_migrator.regression.random_forest_reg import (
    serialize_random_forest_reg,
    deserialize_random_forest_reg
)

from sklearn_migrator.regression.gradient_boosting_reg import (
    serialize_gradient_boosting_reg,
    deserialize_gradient_boosting_reg
)
```

### 🧮 Classification Models

```python
from sklearn_migrator.classification.decision_tree_clf import (
    serialize_decision_tree_clf,
    deserialize_decision_tree_clf
)

from sklearn_migrator.classification.logistic_regression_clf import (
    serialize_logistic_regression_clf,
    deserialize_logistic_regression_clf
)

from sklearn_migrator.classification.random_forest_clf import (
    serialize_random_forest_clf,
    deserialize_random_forest_clf
)

from sklearn_migrator.classification.gradient_boosting_clf import (
    serialize_gradient_boosting_clf,
    deserialize_gradient_boosting_clf
)
```

---

## 🔧 Development

### Run tests

```bash
pytest tests/
```

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
MLOps Engineer | Open Source Contributor
GitHub: [@anvaldes](https://github.com/anvaldes)

---
