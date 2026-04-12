import json
import sklearn
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn_migrator.regression.mlp_reg import serialize_mlp_reg

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

X_train_reg = pd.read_csv('/data/X_train_reg.csv')
y_train_reg = pd.read_csv('/data/y_train_reg.csv')
X_test_reg  = pd.read_csv('/data/X_test_reg.csv')

model = MLPRegressor()
model.fit(X_train_reg, y_train_reg)

y_pred = pd.DataFrame(model.predict(X_test_reg))
y_pred.to_csv('/input/y_pred_input.csv', index=False)

serialized_model = serialize_mlp_reg(model, version_sklearn_in)
with open("/input/serialized_model.json", "w") as f:
    json.dump(serialized_model, f, default=convert)