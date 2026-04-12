import json
import sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn_migrator.regression.lasso_reg import deserialize_lasso_reg

version_sklearn_out = sklearn.__version__

X_test_reg = pd.read_csv('/data/X_test_reg.csv')

with open("/output/serialized_model.json", "r") as f:
    serialized_model = json.load(f)

deserialized_model = deserialize_lasso_reg(serialized_model, version_sklearn_out)

y_pred = pd.DataFrame(deserialized_model.predict(X_test_reg))
y_pred.to_csv('/output/y_pred_output.csv', index=False)