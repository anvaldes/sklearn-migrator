import sklearn
import numpy as np
from sklearn.cluster import KMeans
from sklearn_migrator.clustering.k_means import serialize_k_means
from sklearn_migrator.clustering.k_means import deserialize_k_means

def test_pca():

    X = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0]
    ])

    model = KMeans(n_clusters=2)
    model.fit(X)

    version = sklearn.__version__
    result = serialize_k_means(model, version_in=version)
    new_model = deserialize_k_means(result, version_out=version)

    assert isinstance(result, dict)

    assert "version_sklearn_in" in result

    assert isinstance(new_model, KMeans)
