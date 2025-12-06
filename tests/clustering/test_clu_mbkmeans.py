import sklearn
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn_migrator.clustering.mini_batch_k_means import serialize_mini_batch_kmeans
from sklearn_migrator.clustering.mini_batch_k_means import deserialize_mini_batch_kmeans

def test_clu_mbkmeans():

    X = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0]
    ])

    model = MiniBatchKMeans(n_clusters=2)
    model.fit(X)

    version = sklearn.__version__
    result = serialize_mini_batch_kmeans(model, version_in=version)
    new_model = deserialize_mini_batch_kmeans(result, version_out=version)

    assert isinstance(result, dict)

    assert "version_sklearn_in" in result

    assert isinstance(new_model, MiniBatchKMeans)
