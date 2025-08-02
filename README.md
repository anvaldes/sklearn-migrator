# sklearn-migrator 🧪

**A Python library to serialize and migrate scikit-learn models across incompatible versions.**

[![PyPI version](https://badge.fury.io/py/sklearn-migrator.svg)](https://pypi.org/project/sklearn-migrator/)

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

You can serialized the model from a environment with a scikit-learn version (for example '1.5.0') and then you can deserialized the model from another environment with a other version (for exmaple '1.7.0').

The deserialized model has the version of the environment where you deserialized it. In this case '1.7.0'.

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

## 2. Using Docker

As you can see in this scenario is very useful work with Docker the create particular environments. To help you we created 30 examples of Dockerfile for the input (with all the 30 differents versions of scikit-learn) and another 30 examples of Dockerfile for the output (again with all the 30 differents versions of scikit-learn).

The main idea here is create a python script (input.py) to wrap the process of serialize the mmodel and other python script (output.py) to wrap the process of deserialize the model. 

LINK

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
