import json
import sklearn
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn_migrator import deserialize

X_clu = pd.read_csv('/data/X_clu.csv')

with open("/output/serialized_model.json", "r") as f:
    serialized_model = json.load(f)

deserialized_model = deserialize(serialized_model)

y_pred = pd.DataFrame(deserialized_model.fit_predict(X_clu))
y_pred.to_csv('/output/y_pred_output.csv', index=False)