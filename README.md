# sklearn-migrator 🧪

**A Python library to serialize and migrate scikit-learn models across incompatible versions.**

[![PyPI version](https://badge.fury.io/py/sklearn-migrator.svg)](https://pypi.org/project/sklearn-migrator/)
[![Tests](https://github.com/yourusername/sklearn-migrator/actions/workflows/test.yml/badge.svg)](https://github.com/yourusername/sklearn-migrator/actions)
[![License](https://img.shields.io/github/license/yourusername/sklearn-migrator.svg)](LICENSE)

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

## 💥 Use Cases

* **Long-term model storage**: Store models in a future-proof format across teams and systems.
* **Production model migration**: Move models safely across major `scikit-learn` upgrades.
* **Auditing and inspection**: Read serialized models as JSON, inspect structure, hyperparameters, and internals.
* **Cross-platform inference**: Serialize in Python, serve elsewhere (e.g., microservices).

---

## 📂 Installation

```bash
pip install sklearn-migrator
```

---

## 🚧 Example

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn_migrator.classification.random_forest_clf import (
    serialize_random_forest_clf, deserialize_random_forest_clf
)

model = RandomForestClassifier().fit(X_train, y_train)

# Serialize to dict
data = serialize_random_forest_clf(model, version_in="1.0.2")

# Save to JSON
import json
with open("model.json", "w") as f:
    json.dump(data, f)

# Load and deserialize
with open("model.json") as f:
    model_dict = json.load(f)

new_model = deserialize_random_forest_clf(model_dict, version_out="1.4.2")

# Use the model
preds = new_model.predict(X_test)
```

---

## 🔧 Development

### Run tests

```bash
pytest tests/
```

### Linting

```bash
black .
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

## 📊 Citation

Coming soon: Paper under preparation for JOSS submission.
