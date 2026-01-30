---
title: "sklearn-migrator: Cross-version migration of scikit-learn models for reproducible MLOps"
authors:
  - name: Alberto Andres Valdes Gonzalez
    orcid: 0009-0000-0752-8519
    affiliation: 1
affiliations:
  - name: Independent Researcher (Chile)
    index: 1
date: 2025-12-12
bibliography: paper.bib
tags:
  - Python
  - machine learning
  - scikit-learn
  - MLOps
  - model reproducibility
  - model migration
  - model persistence
---

# Summary

`sklearn-migrator` is a Python library for **safely migrating scikit-learn models across versions** while preserving inference behavior and remaining robust to internal attribute changes. scikit-learn is among the most widely used machine learning libraries in both research and industry, and its estimators are commonly deployed in tabular-data domains such as finance, risk, operations, and marketing [@pedregosa2011; @kaggle2021]. In these settings, model upgrades often coincide with dependency upgrades, container base-image updates, or security patching cycles—making version-to-version portability a practical requirement for production MLOps.

The core problem is that standard persistence mechanisms (pickle/joblib) are **version-fragile**: models saved under one scikit-learn release may fail to load—or may load with altered behavior—under another. This limitation is explicitly cautioned in the official documentation and has been observed in empirical and experiential work [@sklearn_persistence; @fitzpatrick2024davos; @parida2025exportformats]. `sklearn-migrator` addresses this gap by exporting supported estimators into **portable, JSON-compatible Python dictionaries** and reconstructing them in a different environment running a target scikit-learn version. The resulting payloads are readable, inspectable, and transportable across environments and teams, enabling long-term reproducibility and governance.

The migration proceeds in two stages. **Stage 1 (parity):** the library captures a minimal set of prediction-critical attributes (constructor parameters and other parity-relevant settings) that guarantee parity of outputs across versions; these are used to reconstruct an equivalent estimator in the target version and validated under a strict tolerance (e.g., `max |y_in - y_out| < 1e-2`). **Stage 2 (compatibility):** remaining attributes are serialized with a version-aware policy that gracefully handles additions, removals, and renames so deserialization does not break across releases. This strategy is designed around the practical reality that estimator internals, defaults, and attribute names shift over time—even when the public API remains stable [@sklearn_persistence; @parida2025exportformats].

Concretely, consider version `0.21.3` exposing `param_1, param_2, param_3, param_4` and version `1.7.0` exposing `param_1, param_2, param_3, param_5`. We use `param_1` and `param_2` to enforce identical predictions across versions. Since `param_3` exists in both versions, it is stored in the payload and reassigned on load. For version-specific attributes, the serializer records values for both `param_4` and `param_5`: in `0.21.3`, the payload stores the real value of `param_4` and the **default** value that `param_5` would take in newer versions; in `1.7.0`, it stores the real value of `param_5` and the default value that `param_4` would take in older versions. During deserialization, attributes are assigned using a simple `try/except` (or `hasattr`) so only valid attributes for the target version are set, and missing ones are safely skipped. This design makes migrations resilient to evolutionary changes such as renamed fields (e.g., `affinity → metric`), reorganized tree/boosting internals, and added default parameters across releases.

This submission supports **21 models** across supervised and unsupervised learning:

- **Classification (7):** `DecisionTreeClassifier`, `RandomForestClassifier`, `GradientBoostingClassifier`, `LogisticRegression`, `KNeighborsClassifier`, `SVC`, `MLPClassifier`
- **Regression (10):** `DecisionTreeRegressor`, `RandomForestRegressor`, `GradientBoostingRegressor`, `LinearRegression`, `Ridge`, `Lasso`, `KNeighborsRegressor`, `SVR`, `AdaBoostRegressor`, `MLPRegressor`
- **Clustering (3):** `AgglomerativeClustering`, `KMeans`, `MiniBatchKMeans`
- **Dimensionality reduction (1):** `PCA`

Collectively, these estimators represent a large fraction of classical ML model families used in practice and commonly reported in practitioner surveys and applied workflows [@kaggle2021]. `sklearn-migrator` has been validated across **32 scikit-learn versions** (`0.21.3 → 1.7.2`), covering **1,024 migration pairs**, with automated, environment-isolated testing and strict parity checks. Continuous integration, unit tests, and an MIT license support reproducibility and adoption in production MLOps workflows.

# Statement of need

Persisted scikit-learn models frequently break across library upgrades because internal attributes, defaults, and serialization details change over time. Standard persistence mechanisms (e.g., pickle/joblib) are **version-fragile**: a model saved under one release may fail to load—or load with altered behavior—under another. This is explicitly cautioned in the official documentation and has been observed in empirical and experiential studies [@sklearn_persistence; @fitzpatrick2024davos; @parida2025exportformats]. The resulting brittleness complicates production upgrades, environment migrations, cross-team sharing, and long-term reproducibility—especially in regulated or audit-heavy contexts where models must remain verifiable over time.

In practice, this fragility creates failure modes that are costly and hard to debug. A dependency upgrade may break deserialization of a mission-critical model artifact. Conversely, pinning old versions indefinitely increases security risk and operational burden. Teams often face an uncomfortable trade-off: **upgrade safely** (but risk breaking legacy models) or **freeze environments** (but accumulate technical debt). This is particularly acute in organizations that operate many model-serving services, notebooks, and batch pipelines—each with slightly different dependency constraints.

Given scikit-learn’s broad adoption in research and industry, there is strong practical value in keeping legacy models usable across releases [@pedregosa2011; @kaggle2021]. The estimator families supported in this submission—tree ensembles, linear/logistic models, nearest neighbors, support vector machines, and MLPs—are among the most commonly taught, prototyped, and deployed methods in applied machine learning. Unsupervised components such as k-means clustering and PCA are similarly ubiquitous in feature engineering and exploratory analysis pipelines.

`sklearn-migrator` addresses this need with a two-stage, version-aware (de)serialization strategy. First, it captures the minimal, prediction-critical attributes to guarantee parity of outputs between versions. Second, it serializes remaining attributes with explicit, version-conditioned defaults so that parameters added, removed, or renamed across releases do not break deserialization. Unlike pickle/joblib, the library uses portable, JSON-compatible Python dictionaries, enabling safe transport, inspection, and storage independent of the original runtime. This design aligns with modern reproducibility needs and with ecosystem efforts focused on lightweight, inspectable artifacts and reproducible computational environments [@fitzpatrick2024davos; @parida2025exportformats].

The library targets practitioners and MLOps teams who must migrate or reproduce models across heterogeneous environments. It supports forward and backward migration and has been exercised across 32 scikit-learn releases (`0.21.3 → 1.7.2`), covering **1,024** version pairs with unit tests and environment-isolated validation. This foundation reduces upgrade risk today while remaining extensible to additional estimators and components in future releases.

# State of the field

Model persistence and portability are longstanding challenges in applied machine learning. In the scikit-learn ecosystem, the officially recommended mechanisms for saving trained estimators—`pickle` and `joblib`—are explicitly documented as *not* guaranteeing forward or backward compatibility across library versions [@sklearn_persistence]. As a result, serialized models are tightly coupled to the exact scikit-learn release and Python environment in which they were created.

Several alternative approaches partially address related concerns. Interoperability frameworks such as ONNX and PMML enable model exchange across runtimes and languages, but they support only a subset of scikit-learn estimators and often sacrifice access to native APIs, custom preprocessing logic, or numerical parity [@parida2025exportformats]. Re-training models after upgrades is a common workaround, but it may be infeasible due to missing data, regulatory constraints, or computational cost.

# Research Impact Statement

This software addresses a critical reproducibility challenge in applied machine learning: the inherent fragility of serialized `scikit-learn` models across library versions. Standard persistence mechanisms (e.g., `pickle`) tie model artifacts to specific environments, creating a "dependency lock-in" that hinders long-term research reproducibility and complicates production MLOps workflows.

By enabling deterministic, cross-version migration, `sklearn-migrator` mitigates the operational risk and technical debt associated with mandatory security patching and dependency upgrades. The library is particularly impactful for high-stakes domains—such as finance, healthcare, and risk modeling—where models must remain verifiable and functional over long lifecycles, and where retraining may be computationally expensive or ethically restricted.

More broadly, this work promotes the transition toward **transparent and inspectable model artifacts**. By moving away from opaque binary formats and focusing on estimator-level compatibility, the tool fills a significant gap in the machine learning ecosystem. It provides a native, Python-centric path to model portability that complements existing interoperability standards while maintaining the full flexibility of the `scikit-learn` API.

# AI Usage Disclosure

Large language models were used to assist with minor grammar checking and phrasing improvements during manuscript preparation. All software design decisions, implementation, experiments, validation, and technical content were authored and verified by the submitting author.

# Design and validation

## Software Design

The design of `sklearn-migrator` follows a modular, estimator-centric architecture that separates model inspection, version-aware serialization, and controlled reconstruction in the target environment. Each supported `scikit-learn` estimator family is implemented as an **isolated migration unit**, allowing fine-grained handling of structural changes across versions without affecting unrelated models.

At a high level, the migration pipeline consists of three functional components:

1. **Introspection Layer**: A lightweight layer that extracts constructor parameters and prediction-critical attributes from a fitted estimator in the source environment. This layer identifies the minimal state required to satisfy the "Stage 1 (parity)" objective.
2. **Version-aware Serialization Layer**: This component encodes attributes into a portable, JSON-compatible Python dictionary. It explicitly records default values for attributes that may not exist in all versions, ensuring the payload remains self-describing and robust to "Stage 2 (compatibility)" challenges.
3. **Deserialization Layer**: Reconstructs an equivalent estimator in the target environment, assigning only attributes that are valid for the destination version and safely skipping or mapping incompatible or deprecated fields.

## Serialization format

Each supported estimator is serialized into a Python dictionary containing:

1. **Metadata**: source version, estimator type, and migration-relevant flags.
2. **Parity-critical reconstruction parameters**: the minimal set of fields required to reconstruct an estimator that produces matching predictions under strict tolerance.
3. **Compatibility attributes**: additional learned attributes and internal fields stored with version-aware rules, including explicit defaults for fields that exist only in some versions.

To keep payloads portable, the library restricts values to JSON-encodable primitives (numbers, strings, booleans, lists, dicts) and encodes arrays using standard Python lists where necessary. This enables storage in plain JSON files, object storage, databases, or artifact registries, and supports inspection and debugging without executing arbitrary code (a common concern with pickle).

## Validation methodology

`sklearn-migrator` validates migrations through:

- **Environment isolation** (e.g., containers) to ensure `version_in` and `version_out` represent real installations.
- **Fixed synthetic datasets** for deterministic evaluation.
- **Strict parity checks** comparing source and migrated predictions under a tolerance (e.g., `1e-4`).

The library has been tested across a full 32×32 version compatibility matrix, totaling **1,024 migration pairs**, and across all supported estimators. This automated validation provides confidence that the two-stage strategy behaves consistently across a large portion of the modern scikit-learn release history.

# Example

The example below trains a `RandomForestRegressor`, serializes it to a portable (JSON-compatible) dictionary, deserializes it, and checks prediction parity within a strict tolerance. In practice, the `version_out` may correspond to a different scikit-learn installation (e.g., upgrading from 1.5.x to 1.7.x).

```python
import json
import numpy as np
import sklearn
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor

from sklearn_migrator.regression.random_forest_reg import (
    serialize_random_forest_reg,
    deserialize_random_forest_reg,
)

# Train in the "source" environment
X, y = make_regression(n_samples=200, n_features=10, random_state=0)
src_version = sklearn.__version__

src_model = RandomForestRegressor(
    n_estimators=50, random_state=0
).fit(X, y)

# Serialize to a portable payload (JSON-compatible dict)
payload = serialize_random_forest_reg(src_model, version_in=src_version)

# (Optional) store as JSON
with open("model.json", "w") as f:
    json.dump(payload, f)

# --- In a different environment, load and deserialize ---
# (For illustration we reuse the same environment; in practice, version_out may differ.)
tgt_version = sklearn.__version__
with open("model.json") as f:
  payload_loaded = json.load(f)

tgt_model = deserialize_random_forest_reg(payload_loaded, version_out=tgt_version)

# Prediction parity check
y_src = src_model.predict(X)
y_tgt = tgt_model.predict(X)
assert np.max(np.abs(y_src - y_tgt)) < 1e-4
print("Prediction parity verified.")
```

Analogous functions exist for all covered estimators (classification, regression, clustering, and dimensionality reduction). In a true two-environment workflow, the serialization code runs in the source environment (e.g., 0.24.2) and the deserialization code runs in the target environment (e.g., 1.7.2); the assertion above remains the same.

# Limitations

- **Two environments required.** End-to-end validation relies on two isolated environments: a source environment (`version_in`) to serialize and a target environment (`version_out`) to deserialize and verify prediction parity. While this can be simulated on one machine, truly isolated setups (e.g., Docker images) are recommended to avoid dependency leakage.
- **Partial scikit-learn coverage.** scikit-learn contains many estimators and pipeline components beyond the 21 currently supported. Additional models (e.g., pipelines, transformers, and further estimators) are not yet supported but are planned for future releases. We actively welcome community contributions to expand coverage.
**Parity tolerance depends on model family.** Some model families may be sensitive to floating-point or solver differences across versions; the library uses strict tolerances by default but these can be adjusted depending on operational requirements.

# Acknowledgements

We thank the scikit-learn core developers and contributors for their open-source work and documentation, as well as the broader NumPy/SciPy/joblib ecosystems on which this project depends. We are grateful to colleagues and early adopters for testing and feedback that shaped the design, and to the JOSS editors and reviewers for guidance during the review process.
