import json
import sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn_migrator.classification.logistic_regression_clf import deserialize_logistic_regression_clf

version_sklearn_out = sklearn.__version__

X_test_clf = pd.read_csv('/data/X_test_clf.csv')

with open("/output/serialized_model.json", "r") as f:
    serialized_model = json.load(f)

deserialized_model = deserialize_logistic_regression_clf(serialized_model, version_sklearn_out)

y_pred = pd.DataFrame(deserialized_model.predict_proba(X_test_clf))
y_pred.to_csv('/output/y_pred_output.csv', index=False)