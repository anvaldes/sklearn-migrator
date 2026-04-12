import json
import sklearn
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn_migrator.dimension.pca import serialize_pca

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

X_dim = pd.read_csv('/data/X_dim.csv')

model = PCA(n_components=2, whiten=True, svd_solver='full')
model.fit(X_dim)

y_pred = pd.DataFrame(model.transform(X_dim))
y_pred.to_csv('/input/y_pred_input.csv', index=False)

serialized_model = serialize_pca(model, version_sklearn_in)
with open("/input/serialized_model.json", "w") as f:
    json.dump(serialized_model, f, default=convert)