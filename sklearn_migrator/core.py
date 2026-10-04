import importlib
import sklearn

# Supported models: sklearn class name -> (module, serialize function, deserialize function).
# The modules are imported lazily, only when a model of that class is migrated.
_REGISTRY = {
    # Classification
    'DecisionTreeClassifier': ('classification.decision_tree_clf', 'serialize_decision_tree_clf', 'deserialize_decision_tree_clf'),
    'RandomForestClassifier': ('classification.random_forest_clf', 'serialize_random_forest_clf', 'deserialize_random_forest_clf'),
    'GradientBoostingClassifier': ('classification.gradient_boosting_clf', 'serialize_gradient_boosting_clf', 'deserialize_gradient_boosting_clf'),
    'LogisticRegression': ('classification.logistic_regression_clf', 'serialize_logistic_regression_clf', 'deserialize_logistic_regression_clf'),
    'KNeighborsClassifier': ('classification.knn_clf', 'serialize_knn_clf', 'deserialize_knn_clf'),
    'SVC': ('classification.svm_clf', 'serialize_svc', 'deserialize_svc'),
    'MLPClassifier': ('classification.mlp_clf', 'serialize_mlp_clf', 'deserialize_mlp_clf'),
    # Regression
    'DecisionTreeRegressor': ('regression.decision_tree_reg', 'serialize_decision_tree_reg', 'deserialize_decision_tree_reg'),
    'RandomForestRegressor': ('regression.random_forest_reg', 'serialize_random_forest_reg', 'deserialize_random_forest_reg'),
    'GradientBoostingRegressor': ('regression.gradient_boosting_reg', 'serialize_gradient_boosting_reg', 'deserialize_gradient_boosting_reg'),
    'LinearRegression': ('regression.linear_regression_reg', 'serialize_linear_regression_reg', 'deserialize_linear_regression_reg'),
    'Ridge': ('regression.ridge_reg', 'serialize_ridge_reg', 'deserialize_ridge_reg'),
    'Lasso': ('regression.lasso_reg', 'serialize_lasso_reg', 'deserialize_lasso_reg'),
    'KNeighborsRegressor': ('regression.knn_reg', 'serialize_knn_reg', 'deserialize_knn_reg'),
    'SVR': ('regression.svm_reg', 'serialize_svr', 'deserialize_svr'),
    'AdaBoostRegressor': ('regression.adaboost_reg', 'serialize_adaboost_reg', 'deserialize_adaboost_reg'),
    'MLPRegressor': ('regression.mlp_reg', 'serialize_mlp_reg', 'deserialize_mlp_reg'),
    # Clustering
    'KMeans': ('clustering.k_means', 'serialize_k_means', 'deserialize_k_means'),
    'MiniBatchKMeans': ('clustering.mini_batch_k_means', 'serialize_mini_batch_kmeans', 'deserialize_mini_batch_kmeans'),
    'AgglomerativeClustering': ('clustering.agglomerative', 'serialize_agglomerative', 'deserialize_agglomerative'),
    # Dimensionality reduction
    'PCA': ('dimension.pca', 'serialize_pca', 'deserialize_pca'),
}

# Key under which serialize() stores the model type in the serialized dictionary.
MODEL_TYPE_KEY = 'model_type'


def supported_models() -> list:
    """
    Return the names of the scikit-learn models supported by the library.

    Returns
    -------
    list
        Sorted list of scikit-learn class names (e.g. 'RandomForestClassifier').
    """

    return sorted(_REGISTRY)


def _load_function(model_type: str, position: int):
    """
    Import and return the serialize (position 1) or deserialize (position 2)
    function registered for a model type.
    """

    entry = _REGISTRY[model_type]
    module = importlib.import_module('.' + entry[0], package=__package__)

    return getattr(module, entry[position])


def serialize(model, version_in: str = None) -> dict:
    """
    Serialize any supported fitted scikit-learn model into a JSON-compatible
    dictionary. The right serializer is selected from the class of the model.

    Parameters
    ----------
    model : object
        A fitted scikit-learn model of one of the supported classes
        (see supported_models()).
    version_in : str, optional
        The sklearn version used to train the model (e.g. '1.2.0').
        Defaults to the installed sklearn version.

    Returns
    -------
    dict
        A dictionary containing all necessary data to reconstruct the model,
        including its type under the 'model_type' key.

    Raises
    ------
    TypeError
        If the model is not an instance of a supported scikit-learn class.
    """

    model_class = type(model)
    model_type = model_class.__name__

    is_sklearn_model = model_class.__module__.split('.')[0] == 'sklearn'

    if not is_sklearn_model or model_type not in _REGISTRY:
        raise TypeError(
            f"Model of type '{model_class.__module__}.{model_type}' is not supported. "
            f"Supported models: {', '.join(supported_models())}."
        )

    if version_in is None:
        version_in = sklearn.__version__

    data = _load_function(model_type, 1)(model, version_in)
    data[MODEL_TYPE_KEY] = model_type

    return data


def deserialize(data: dict, version_out: str = None, model_type: str = None):
    """
    Reconstruct a model from a dictionary produced by serialize(). The right
    deserializer is selected from the model type stored in the dictionary.

    Parameters
    ----------
    data : dict
        Dictionary produced by serialize().
    version_out : str, optional
        The sklearn version of the target environment (e.g. '1.7.0').
        Defaults to the installed sklearn version.
    model_type : str, optional
        Name of the scikit-learn class of the model (e.g. 'RandomForestClassifier').
        Only needed for dictionaries that do not store their model type, such as
        those produced by the model-specific serialize_<model> functions.

    Returns
    -------
    object
        The reconstructed model, compatible with the target environment.

    Raises
    ------
    ValueError
        If the model type is missing from the dictionary or is not supported.
    """

    if model_type is None:
        model_type = data.get(MODEL_TYPE_KEY)

    if model_type is None:
        raise ValueError(
            f"The serialized dictionary has no '{MODEL_TYPE_KEY}' field, so the model "
            "cannot be identified. Pass it explicitly, e.g. "
            "deserialize(data, model_type='RandomForestClassifier')."
        )

    if model_type not in _REGISTRY:
        raise ValueError(
            f"Model type '{model_type}' is not supported. "
            f"Supported models: {', '.join(supported_models())}."
        )

    if version_out is None:
        version_out = sklearn.__version__

    return _load_function(model_type, 2)(data, version_out)
