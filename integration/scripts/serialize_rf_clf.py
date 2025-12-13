import json
import argparse
import numpy as np
import sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn_migrator.classification.random_forest_clf import serialize_random_forest_clf


def to_jsonable(x):
    """
    Recursively convert numpy / sklearn objects to JSON-serializable types.
    """
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_jsonable(v) for v in x]
    if isinstance(x, tuple):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-model", required=True)
    ap.add_argument("--out-pred", required=True)
    args = ap.parse_args()

    rng = np.random.RandomState(0)
    X = rng.randn(200, 5)
    y = (X[:, 0] > 0).astype(int)

    model = RandomForestClassifier(n_estimators=30, random_state=42)
    model.fit(X, y)

    version_in = sklearn.__version__
    metadata = serialize_random_forest_clf(model, version_in=version_in)

    with open(args.out_model, "w") as f:
        json.dump(to_jsonable(metadata), f)

    y_pred = model.predict(X).astype(int)
    np.save(args.out_pred, y_pred)


if __name__ == "__main__":
    main()

