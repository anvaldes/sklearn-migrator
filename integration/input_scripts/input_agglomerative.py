import json
import sklearn
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn_migrator.clustering.agglomerative import serialize_agglomerative

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

X_clu = pd.read_csv('/data/X_clu.csv')

model = AgglomerativeClustering(n_clusters=4)
model.fit(X_clu)

y_pred = pd.DataFrame(model.fit_predict(X_clu))
y_pred.to_csv('/input/y_pred_input.csv', index=False)

serialized_model = serialize_agglomerative(model, version_sklearn_in)
with open("/input/serialized_model.json", "w") as f:
    json.dump(serialized_model, f, default=convert)