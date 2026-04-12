import json
import sklearn
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn_migrator.clustering.mini_batch_k_means import serialize_mini_batch_kmeans

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

model = MiniBatchKMeans(n_clusters=3, random_state=0)
model.fit(X_clu)

y_pred = pd.DataFrame(model.predict(X_clu))
y_pred.to_csv('/input/y_pred_input.csv', index=False)

serialized_model = serialize_mini_batch_kmeans(model, version_sklearn_in)
with open("/input/serialized_model.json", "w") as f:
    json.dump(serialized_model, f, default=convert)