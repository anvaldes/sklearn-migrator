import json
import argparse
import numpy as np
import sklearn

from sklearn_migrator.classification.random_forest_clf import deserialize_random_forest_clf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-model", required=True)
    ap.add_argument("--out-pred", required=True)
    args = ap.parse_args()

    rng = np.random.RandomState(0)
    X = rng.randn(200, 5)

    with open(args.in_model, "r") as f:
        metadata = json.load(f)

    version_out = sklearn.__version__
    new_model = deserialize_random_forest_clf(metadata, version_out=version_out)
    new_model.classes_ = np.array(new_model.classes_)

    y_pred_new = new_model.predict(X).astype(int)
    np.save(args.out_pred, y_pred_new)


if __name__ == "__main__":
    main()
