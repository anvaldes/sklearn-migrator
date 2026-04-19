import warnings
import numpy as np
from sklearn.tree._tree import Tree
from sklearn.tree import DecisionTreeClassifier

all_features = [
    'max_leaf_nodes',
    'min_samples_split',
    'n_classes_',
    'splitter',
    'min_impurity_decrease',
    'max_features_',
    'min_weight_fraction_leaf',
    'min_samples_leaf',
    'max_features',
    'class_weight',
    'max_depth',
    'classes_',
    'n_outputs_',
    'criterion',
    'presort',
    'min_impurity_split',
    'ccp_alpha',
    'feature_names_in_',
    'monotonic_cst'
]


def version_tuple(version: str) -> tuple:

    """
    Convert a version string into a comparable tuple of integers.

    Parameters
    ----------
    version : str
        Version string (e.g. '1.2.0').

    Returns
    -------
    tuple
        Tuple of integers (major, minor, patch).
    """

    version_split = version.split('.')

    if len(version_split) == 1:
        new_version = (int(version_split[0]), 0, 0)
    elif len(version_split) == 2:
        new_version = (int(version_split[0]), int(version_split[1]), 0)
    elif len(version_split) == 3:
        new_version = (int(version_split[0]), int(version_split[1]), int(version_split[2]))
    else:
        new_version = 'Formato no valido'

    return new_version


def _get_extended_nodes(nodes: list, version_in: str) -> list:
    """
    Extend node tuples with a placeholder field for sklearn versions < 1.3.

    Parameters
    ----------
    nodes : list
        List of node tuples from the tree state.
    version_in : str
        The sklearn version used to train the model.

    Returns
    -------
    list
        List of node tuples, extended if necessary.
    """

    if (version_tuple('0.21.3') <= version_tuple(version_in)) and (version_tuple(version_in) < version_tuple('1.3')):
        return [node + (0,) for node in nodes]
    return nodes


def _build_dtype_dict(dtypes: np.dtype, version_in: str) -> dict:
    """
    Build a dictionary describing the node dtype structure of the tree.

    Parameters
    ----------
    dtypes : np.dtype
        The dtype of the nodes array from the tree state.
    version_in : str
        The sklearn version used to train the model.

    Returns
    -------
    dict
        Dictionary with field names, formats, offsets and itemsize.
    """


    field_names = dtypes.names
    formats = [dtypes.fields[name][0] for name in field_names]
    offsets = [dtypes.fields[name][1] for name in field_names]
    itemsize = dtypes.itemsize

    if (version_tuple('0.21.3') <= version_tuple(version_in)) and (version_tuple(version_in) < version_tuple('1.3')):
        return {
            'field_names': list(field_names + ('missing_go_to_left',)),
            'formats': [str(fmt) for fmt in formats + [np.dtype('uint8')]],
            'offsets': [int(off) for off in offsets + [56]],
            'itemsize': 64
        }

    return {
        'field_names': list(field_names),
        'formats': [str(fmt) for fmt in formats],
        'offsets': [int(off) for off in offsets],
        'itemsize': int(itemsize)
    }


def _get_metadata(model: DecisionTreeClassifier, version_in: str) -> dict:
    """
    Extract feature metadata from a fitted DecisionTreeClassifier.

    Parameters
    ----------
    model : DecisionTreeClassifier
        A fitted scikit-learn DecisionTreeClassifier instance.
    version_in : str
        The sklearn version used to train the model.

    Returns
    -------
    dict
        Dictionary with n_features_in, n_features, n_classes and n_outputs.
    """

    dict_metadata = {}

    try:
        dict_metadata['n_features_in'] = model.n_features_in_
    except:
        dict_metadata['n_features_in'] = None

    try:
        dict_metadata['n_features'] = model.n_features_
    except:
        dict_metadata['n_features'] = None

    dict_metadata['n_classes'] = model.n_classes_
    dict_metadata['n_outputs'] = model.n_outputs_

    return dict_metadata


def serialize_decision_tree_clf(model: DecisionTreeClassifier, version_in: str) -> dict:
    """
    Serialize a fitted DecisionTreeClassifier into a JSON-compatible dictionary.

    Parameters
    ----------
    model : DecisionTreeClassifier
        A fitted scikit-learn DecisionTreeClassifier instance.
    version_in : str
        The sklearn version used to train the model (e.g. '1.2.0').

    Returns
    -------
    dict
        A dictionary containing all necessary data to reconstruct the model.
    """

    tree = model.tree_
    state = tree.__getstate__()

    serialized_tree = {
        'max_depth': int(state['max_depth']),
        'node_count': int(state['node_count']),
        'values': state['values'].tolist(),
        'nodes': [list(n) for n in _get_extended_nodes(state['nodes'].tolist(), version_in)],
        'dtypes': _build_dtype_dict(state['nodes'].dtype, version_in)
    }

    metadata = _get_metadata(model, version_in)
    metadata['serialized_tree'] = serialized_tree
    metadata['version_sklearn_in'] = version_in

    model_dict = model.__dict__
    model_dict_keys = list(model_dict.keys())

    default_values = {
        'min_impurity_split': None,
        'presort': False,
        'ccp_alpha': 0.0,
        'feature_names_in_': None,
        'monotonic_cst': None
    }

    kdv = list(default_values.keys())

    other_params = {}

    for af in all_features:
        if (af in model_dict_keys) == False:
            other_params[af] = default_values[af]
        else:
            other_params[af] = model_dict[af]

    metadata['other_params'] = other_params

    return metadata

def _build_tree_dtype(dtypes_dict: dict, version_out: str) -> tuple:
    """
    Reconstruct the numpy dtype for the nodes array of the target sklearn version.

    Parameters
    ----------
    dtypes_dict : dict
        Dictionary produced by _build_dtype_dict.
    version_out : str
        The sklearn version of the target environment.

    Returns
    -------
    tuple
        A tuple of (np.dtype, int) with the dtype and number of elements to use.
    """

    version_lt_1_3 = version_tuple(version_out) < version_tuple('1.3')
    num_elements = 7 if version_lt_1_3 else 8

    field_names = dtypes_dict['field_names'][:num_elements]
    formats = [np.dtype(fmt) for fmt in dtypes_dict['formats'][:num_elements]]
    offsets = dtypes_dict['offsets'][:num_elements]
    itemsize = 56 if version_lt_1_3 else 64

    return np.dtype({
        'names': field_names,
        'formats': formats,
        'offsets': offsets,
        'itemsize': itemsize
    }), num_elements


def deserialize_decision_tree_clf(data: dict, version_out: str) -> DecisionTreeClassifier:
    """
    Reconstruct a DecisionTreeClassifier from a serialized dictionary.

    Parameters
    ----------
    data : dict
        Dictionary produced by serialize_decision_tree_clf.
    version_out : str
        The sklearn version of the target environment (e.g. '1.7.0').

    Returns
    -------
    DecisionTreeClassifier
        A reconstructed scikit-learn DecisionTreeClassifier instance.
    """

    version_in = data['version_sklearn_in']
    serialized = data['serialized_tree']
    dtype_dict = serialized['dtypes']

    tree_dtype, num_elements = _build_tree_dtype(dtype_dict, version_out)

    serialized['nodes'] = [tuple(n[:num_elements]) for n in serialized['nodes']]
    nodes_array = np.array(serialized['nodes'], dtype=tree_dtype)
    values_array = np.array(serialized['values'])

    if (version_tuple(version_out) >= version_tuple('1.4')) and (version_tuple(version_in) <= version_tuple('1.3.2')):

        sums = values_array.sum(axis=2, keepdims=True)
        normalized_values = np.divide(values_array, sums)
        values_array = normalized_values

    n_classes = np.array([data['n_classes']], dtype=np.intp)  # classification
    n_outputs = data['n_outputs']
    n_features = (data['n_features'] or data['n_features_in'])

    tree_obj = Tree(n_features, n_classes, n_outputs)
    tree_obj.__setstate__({
        'max_depth': serialized['max_depth'],
        'node_count': serialized['node_count'],
        'nodes': nodes_array,
        'values': values_array
    })

    new_tree = DecisionTreeClassifier(max_depth=serialized['max_depth'], random_state=42)
    new_tree.tree_ = tree_obj
    new_tree.n_outputs_ = n_outputs
    new_tree.n_classes_ = np.int64(data['n_classes'])

    try:
        new_tree.n_features_ = n_features
    except:
        pass

    try:
        new_tree.n_features_in_ = n_features
    except:
        pass

    for af in all_features:
        try:
            new_tree.__dict__[af] = data['other_params'][af]
        except KeyError:
            pass  # field not present in this sklearn version
        except AttributeError:
            pass  # attribute not settable in this sklearn version
        except Exception as e:
            warnings.warn(
                f"Could not set field '{af}' on {type(new_tree).__name__}: "
                f"{type(e).__name__}: {e}. Field will be skipped.",
                UserWarning,
            )
            
    return new_tree

