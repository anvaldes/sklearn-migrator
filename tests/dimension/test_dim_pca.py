import sklearn
import numpy as np
from sklearn.decomposition import PCA
from sklearn_migrator import serialize, deserialize

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
    result = serialize(model, version_in=version)
    new_model = deserialize(result, version_out=version)

    #--------------------------------------------------

    assert isinstance(result, dict)

    assert "version_sklearn_in" in result

    assert isinstance(new_model, PCA)

    #--------------------------------------------------

    vec = model.transform(X)
    new_vec = new_model.transform(X)

    threshold = 1e-2

    assert (abs(new_vec - vec).max(axis = 1).max() <= threshold)

    #--------------------------------------------------
