import json
import pandas as pd
import sklearn
import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn_migrator.regression.random_forest_reg import (
    serialize_random_forest_reg,
)
from sklearn_migrator.utils import json_convert

def test_random_forest_json_serializability_with_pandas():
    """
    Test that RandomForestRegressor serialized with pandas feature names
    is JSON serializable (Issue #141).
    """
    # 1. Create dummy data with explicit string feature names using pandas
    X = pd.DataFrame([[1, 2], [3, 4]], columns=["feature_alpha", "feature_beta"])
    y = [10.0, 20.0]

    # 2. Fit the model to populate model.feature_names_in_ as an ndarray
    model = RandomForestRegressor(n_estimators=2, max_depth=2).fit(X, y)

    # 3. Extract version-aware dictionary
    data = serialize_random_forest_reg(model, sklearn.__version__)

    # 4. Verify JSON dump works
    try:
        json_str = json.dumps(data)
        assert isinstance(json_str, str)
    except TypeError as e:
        pytest.fail(f"Serialization failed with TypeError: {e}")

def test_numpy_scalar_conversion():
    """
    Test that individual numpy scalars are converted to native Python types.
    """
    test_data = {
        'int64': np.int64(42),
        'float64': np.float64(3.14),
        'bool_': np.bool_(True),
        'nested': {
            'arr': np.array([1, 2, 3]),
            'scalar': np.int32(7)
        }
    }
    
    converted = json_convert(test_data)
    
    # Check types
    assert isinstance(converted['int64'], int)
    assert not isinstance(converted['int64'], np.generic)
    
    assert isinstance(converted['float64'], float)
    assert not isinstance(converted['float64'], np.generic)
    
    assert isinstance(converted['bool_'], bool)
    assert not isinstance(converted['bool_'], np.generic)
    
    assert isinstance(converted['nested']['arr'], list)
    assert isinstance(converted['nested']['scalar'], int)
    
    # Ensure it's JSON serializable
    json_str = json.dumps(converted)
    assert isinstance(json_str, str)

def test_all_features_names_in_coercion():
    """
    Verify that if feature_names_in_ is present, it is converted to list in the output.
    """
    X = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "b"])
    y = [0, 1]
    model = RandomForestRegressor(n_estimators=1).fit(X, y)
    
    data = serialize_random_forest_reg(model, sklearn.__version__)
    
    # feature_names_in_ should be in other_params
    feature_names = data['other_params'].get('feature_names_in_')
    assert feature_names is not None
    assert isinstance(feature_names, list)
    assert feature_names == ["a", "b"]
