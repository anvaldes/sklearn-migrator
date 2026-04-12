import json
import sklearn
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn_migrator.dimension.pca import deserialize_pca

version_sklearn_out = sklearn.__version__

X_dim = pd.read_csv('/data/X_dim.csv')

with open("/output/serialized_model.json", "r") as f:
    serialized_model = json.load(f)

deserialized_model = deserialize_pca(serialized_model, version_sklearn_out)

y_pred = pd.DataFrame(deserialized_model.transform(X_dim))
y_pred.to_csv('/output/y_pred_output.csv', index=False)