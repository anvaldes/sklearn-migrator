import json
import joblib
import sklearn
import numpy as np
import pandas as pd
from joblib import load

from sklearn.svm import SVR

from sklearn_migrator.regression.svm_reg import serialize_svr

#--------------------------------------------------

version_sklearn_in = sklearn.__version__

#--------------------------------------------------

# Functions

def convert(o):
    if isinstance(o, (np.integer, np.int64)):
        return int(o)
    elif isinstance(o, (np.floating, np.float64)):
        return float(o)
    elif isinstance(o, np.ndarray):
        return o.tolist()
    else:
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

#--------------------------------------------------

# 1. Load data

X_train_reg = pd.read_csv('/data/X_train_reg.csv')
y_train_reg = pd.read_csv('/data/y_train_reg.csv')
X_test_reg = pd.read_csv('/data/X_test_reg.csv')
y_test_reg = pd.read_csv('/data/y_test_reg.csv')

#--------------------------------------------------

# 2. Training model

model = SVR()
model.fit(X_train_reg, y_train_reg)

#--------------------------------------------------

# 3. Model inference

y_pred = pd.DataFrame(model.predict(X_test_reg))

#--------------------------------------------------

# 4. Save inference

y_pred.to_csv('/input/y_pred_input.csv', index = False)

#--------------------------------------------------

# 5. Save model

serialized_model = serialize_svr(model, version_sklearn_in)

with open("/input/serialized_model.json", "w") as f:
    json.dump(serialized_model, f, default=convert)

#--------------------------------------------------