import numpy as np
from sklearn.decomposition import PCA

all_features = [
    '_fit_svd_solver',
    'components_',
    'copy',
    'explained_variance_',
    'explained_variance_ratio_',
    'iterated_power',
    'mean_',
    'n_components',
    'n_components_',
    'n_samples_',
    'noise_variance_',
    'singular_values_',
    'svd_solver',
    'tol',
    'whiten',
    'feature_names_in_',
    'n_features_',
    'n_features_in_',
    'power_iteration_normalizer'
    ]


def serialize_pca(model, version_in):

    metadata = {}

    init_params = model.get_params()

    try:
        del init_params['n_oversamples']
    except:
        pass

    try:
        del init_params['power_iteration_normalizer']
    except:
        pass

    metadata['init_params'] = init_params

    model_dict = model.__dict__
    model_dict_keys = list(model_dict.keys())

    default_values = {
        'feature_names_in_': None,
        'n_features_': len(model_dict['mean_']),
        'n_features_in_': len(model_dict['mean_']),
        'power_iteration_normalizer': 'auto'
        }

    other_params = {}
    
    for af in all_features:
        if (af in model_dict_keys) == False:
            other_params[af] = default_values[af]
        else:
            other_params[af] = model_dict[af]

    metadata['other_params'] = other_params
    metadata['version_sklearn_in'] = version_in

    return metadata


def deserialize_pca(data, version_out):
    version_in = data['version_sklearn_in']
    init_params = data['init_params']

    # 1. Crear el modelo con los parámetros de inicialización
    new_model = PCA(**init_params)

    # 2. Campos que deben ser numpy arrays
    array_fields = [
        'components_',
        'explained_variance_',
        'explained_variance_ratio_',
        'mean_',
        'singular_values_'
    ]

    other_params = data['other_params']

    for af in all_features:
        if af not in other_params:
            continue

        value = other_params[af]

        # Convertir listas a np.array cuando corresponde
        if af in array_fields and value is not None and not isinstance(value, np.ndarray):
            value = np.array(value)

        # Asignar al __dict__ del modelo
        new_model.__dict__[af] = value

    return new_model