import json
import joblib
import sklearn
import numpy as np
import pandas as pd
from joblib import load

from sklearn.tree import DecisionTreeRegressor

from sklearn_migrator.regression.decision_tree_reg import deserialize_decision_tree_reg

#--------------------------------------------------

version_sklearn_out = sklearn.__version__

#--------------------------------------------------

# 1. Load data

X_train_reg = pd.read_csv('/data/X_train_reg.csv')
y_train_reg = pd.read_csv('/data/y_train_reg.csv')
X_test_reg = pd.read_csv('/data/X_test_reg.csv')
y_test_reg = pd.read_csv('/data/y_test_reg.csv')

#--------------------------------------------------

# 2. Load model

with open("/output/serialized_model.json", "r") as f:
    serialized_model = json.load(f)

#--------------------------------------------------

# 3. Deserialization

deserialized_model = deserialize_decision_tree_reg(serialized_model, version_sklearn_out)

#--------------------------------------------------

# 4. Inference with deserialized_model

y_pred = pd.DataFrame(deserialized_model.predict(X_test_reg))

#--------------------------------------------------

# 5. Save inference

y_pred.to_csv('/output/y_pred_output.csv', index = False)

#--------------------------------------------------