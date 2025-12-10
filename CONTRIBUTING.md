# Contributing to `sklearn-migrator`

Thanks for your interest in contributing! This guide explains how to develop, test, and propose changes—especially how to add **robust serialization/deserialization** support for scikit-learn models across versions.

> Please open an Issue first for significant changes (new estimator, major refactor) to align on scope and approach.
> Do **not** bump the package version in your PR. Maintainers handle releases and tags.

---

## Quick Start

**Environment**
- Python 3.9–3.12 (matches CI).
- Minimal workflow:
  ```bash
  pip install -r requirements-dev.txt
  pip install -e .
  pytest
  ```

**Lint & Format** (optional, recommended)
```bash
pre-commit install
pre-commit run -a
```

**Project Layout**
```
sklearn_migrator/
  classification/
    <estimator>_clf.py
  regression/
    <estimator>_reg.py
tests/
  classification/test_<estimator>_clf.py
  regression/test_<estimator>_reg.py
```

---

## Goal of a Migration

We consider a migration **correct** when a model serialized in one scikit-learn version can be **deserialized** in another version and **produces the same predictions** (up to a small numerical tolerance) on the same input.

Acceptance check:
```text
max_abs_diff = max(abs(y_pred_input - y_pred_output)) < 1e-4
```

---

## Supported / Targeted scikit-learn Versions

We commonly validate against the versions below (extend if needed):

```
0.21.3
0.22.0
0.22.1
0.23.0
0.23.1
0.23.2
0.24.0
0.24.1
0.24.2
1.0.0
1.0.1
1.0.2
1.1.0
1.1.1
1.1.2
1.1.3
1.2.0
1.2.1
1.2.2
1.3.0
1.3.1
1.3.2
1.4.0
1.4.2
1.5.0
1.5.1
1.5.2
1.6.0
1.6.1
1.7.0
1.7.1
1.7.2
```

You can leverage prebuilt environment recipes here:  
**https://github.com/anvaldes/environments_scikit_learn**

---

## Adding a New Estimator (Serialization + Deserialization)

Below is a suggested workflow using **Docker-based per-version environments** to extract parameters and validate cross-version behavior. As a running example, consider `DecisionTreeRegressor`.

### 1) Extract Model Attributes per Version

For each scikit-learn version:
1. Train a tiny model (deterministic data, fixed random_state).
2. Inspect **all attributes** (not only `get_params()`):

```python
model_dict = model.__dict__
model_dict_keys = list(model_dict.keys())
```

Persist `model_dict_keys` per version (e.g., JSON/CSV). This lets you compute intersections/differences across versions.

### 2) Identify **Constructor Parameters**

- **Constructor parameters** are the minimal set of attributes that:
  1) exist **in every version** for the estimator, **and**
  2) are **directly used for prediction**.

Examples:
- LinearRegression → `coef_`, `intercept_`.
- Tree-based models → internal `tree_` payload plus any attributes essential for inference.

These are the backbone of the serializer—ensure these are always captured and used to rebuild a functionally equivalent model.

### 3) Handle Non-Constructor Parameters

Classify the rest:
- **Present in all versions – non-constructor** → store in a simple dict.
- **Not present in all versions** → use **version-aware defaults**:

**Example rule**:
- `var_1` exists only in versions `< 1.1`
- `var_2` exists only in versions `>= 1.1`
- `par_1`, `par_2`, `par_3` exist in all versions

**Serialization**:
1) Always store `par_1`, `par_2`, `par_3`.
2) For `< 1.1`: store `var_1`, and store `var_2` as the **default value** used in `>= 1.1`.
3) For `>= 1.1`: store `var_2`, and store `var_1` as the **default value** used in `< 1.1`.

**Deserialization**:
- Use `try/except` or `hasattr()` to set only attributes that exist in the **target** version; fall back to the stored default otherwise.

> It’s acceptable to **skip 1–2 problematic parameters** if they add undue complexity. Keep exceptions minimal and add clear code comments.

### 4) Cross-Version Validation with Docker

Use one container per version. A minimal driver can look like:

```python
# input.py (runs in version v_input)
# 1) train a small model
# 2) predict and save y_pred_input
# 3) serialize model (e.g., model.json) → /input/
# -> writes /input/y_pred_input.csv, /input/model.json

# output.py (runs in version v_output)
# 1) load /input/model.json
# 2) deserialize into a fresh estimator
# 3) predict and save y_pred_output
# 4) compare: max(abs(y_pred_input - y_pred_output)) < 1e-4
```

Example orchestration snippet:
```python
import subprocess, os

image_name_input = "sklearn_input_image"
local_path = "/ABSOLUTE/PATH/TO/workdir"

subprocess.run([
    "docker", "build",
    "-f", "Dockerfile_input",
    "-t", image_name_input,
    "."
], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("Created INPUT image ✅\n")

subprocess.run([
    "docker", "run", "--rm",
    "-v", f"{local_path}/input:/input",
    image_name_input
], check=True)

print("Ran INPUT ✅\n")
```

Repeat similarly for the **output** container. Then loop over the Cartesian product of `(v_input, v_output)` and assert prediction parity.

> You can reuse and adapt the ready-to-go images from **`environments_scikit_learn`** to avoid manual environment setup.

### 5) Where to Place Code & Tests

Follow the existing structure:

- **Code**
  - `sklearn_migrator/regression/decision_tree_reg.py`
  - `sklearn_migrator/classification/decision_tree_clf.py`
- **Unit tests**
  - `tests/regression/test_decision_tree_reg.py`
  - `tests/classification/test_decision_tree_clf.py`

Browse existing implementations for guidance:  
- **https://github.com/anvaldes/sklearn-migrator/tree/dev/sklearn_migrator**  
- **https://github.com/anvaldes/sklearn-migrator/tree/dev/tests**

---

## Testing Guidelines

- Keep tests **small, fast, deterministic** (tiny synthetic datasets).
- Validate **prediction parity** across at least a few representative version pairs (e.g., oldest ↔ newest; around known breaking changes).
- Follow existing test patterns in `/tests`.

Minimal example:
```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn_migrator.regression.decision_tree_reg import serialize, deserialize

def test_decision_tree_reg_roundtrip():
    X = np.array([[0.0],[1.0],[2.0]], dtype=float)
    y = np.array([0.0, 1.0, 4.0], dtype=float)

    model = DecisionTreeRegressor(random_state=0).fit(X, y)
    payload = serialize(model)

    model2 = deserialize(payload)
    y1 = model.predict(X)
    y2 = model2.predict(X)

    assert np.max(np.abs(y1 - y2)) < 1e-4
```
---

## Dev Tooling

**Install / Test**
```bash
pip install -r requirements-dev.txt
pip install -e .
pytest
```

**Coverage** (if configured in `pytest.ini`):
```bash
pytest --cov=sklearn_migrator --cov-report=xml
```

**Lint & Format** (pre-commit with ruff/black)
```bash
pre-commit install
pre-commit run -a
```

---

## Pull Request Checklist

- [ ] Code implements serializer/deserializer following the version-aware strategy.
- [ ] Tests added/updated (`tests/...`) and pass locally (`pytest`).
- [ ] Cross-version parity validated (provide a short summary in the PR).
- [ ] Lint/format clean (ruff/black or `pre-commit run -a`).
- [ ] No version bump (maintainers will handle releases/tags).
- [ ] Clear docstrings/comments for any version-specific defaults or skipped params.

**Branching**  
Create from `dev` with a descriptive name, e.g.:
```
feature/decision-tree-reg
fix/logreg-defaults-<1.1
```

**Commit style** (suggested):
```
feat(tree): add DecisionTreeRegressor serializer/deserializer
fix(logreg): handle missing class_weight for <1.1 with default mapping
test: add cross-version parity tests for 0.24.2 <-> 1.3.2
docs: explain constructor parameters for LinearRegression
```

---

## License

By contributing, you agree that your contributions will be licensed under the **MIT License** of this repository.

---

## Thanks!

Your help makes `sklearn-migrator` more reliable and useful for the community. If you have questions, please open an Issue or start a Discussion—we’re happy to help!
