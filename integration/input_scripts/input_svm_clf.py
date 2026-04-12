import json
import sklearn
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn_migrator.classification.svm_clf import serialize_svc

version_sklearn_in = sklearn.__version__

def convert(o):
    if isinstance(o, (np.integer, np.int64)):
        return int(o)
    elif isinstance(o, (np.floating, np.float64)):
        return float(o)
    elif isinstance(o, np.ndarray):
        return o.tolist()
    else:
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

X_train_clf = pd.read_csv('/data/X_train_clf.csv')
y_train_clf = pd.read_csv('/data/y_train_clf.csv')
X_test_clf  = pd.read_csv('/data/X_test_clf.csv')

model = SVC(probability=True)
model.fit(X_train_clf, y_train_clf)

y_pred = pd.DataFrame(model.predict_proba(X_test_clf))
y_pred.to_csv('/input/y_pred_input.csv', index=False)

serialized_model = serialize_svc(model, version_sklearn_in)
with open("/input/serialized_model.json", "w") as f:
    json.dump(serialized_model, f, default=convert)