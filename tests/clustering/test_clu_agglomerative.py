import sklearn
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn_migrator.clustering.agglomerative import serialize_agglomerative
from sklearn_migrator.clustering.agglomerative import deserialize_agglomerative

def test_clu_agglomerative():

    X = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0]
    ])

    model = AgglomerativeClustering(n_clusters=2)
    model.fit(X)

    version = sklearn.__version__
    result = serialize_agglomerative(model, version_in=version)
    new_model = deserialize_agglomerative(result, version_out=version)

    #--------------------------------------------------

    assert isinstance(result, dict)

    assert "version_sklearn_in" in result

    assert isinstance(new_model, AgglomerativeClustering)

    #--------------------------------------------------

    labels_original = model.labels_.copy()

    new_model.fit(X)
    labels_migrated = new_model.labels_

    # Dependiendo de tu implementación, esto puede ser exacto:
    assert labels_original.shape == labels_migrated.shape
    assert np.array_equal(labels_original, labels_migrated)

    #--------------------------------------------------
