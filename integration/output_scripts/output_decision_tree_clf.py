import json
import sklearn
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn_migrator import deserialize

X_test_clf = pd.read_csv('/data/X_test_clf.csv')

with open("/output/serialized_model.json", "r") as f:
    serialized_model = json.load(f)

deserialized_model = deserialize(serialized_model)

y_pred = pd.DataFrame(deserialized_model.predict_proba(X_test_clf))
y_pred.to_csv('/output/y_pred_output.csv', index=False)