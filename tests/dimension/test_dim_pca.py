import sklearn
import numpy as np
from sklearn.decomposition import PCA
from sklearn_migrator.dim_reduction.pca import serialize_pca
from sklearn_migrator.dim_reduction.pca import deserialize_pca

def test_pca():

    X = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0]
    ])

    model = PCA(n_components=2)
    model.fit(X)

    version = sklearn.__version__
    result = serialize_pca(model, version_in=version)
    new_model = deserialize_pca(result, version_out=version)

    assert isinstance(result, dict)

    assert "version_sklearn_in" in result

    assert isinstance(new_model, PCA)
