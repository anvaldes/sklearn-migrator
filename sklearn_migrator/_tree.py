import numpy as np
from sklearn.tree._tree import Tree
from .utils import version_tuple


def _lacks_missing_go_to_left(version: str) -> bool:
    """
    Tell whether the tree nodes of a sklearn version lack the
    'missing_go_to_left' field (added in sklearn 1.3).
    """

    return version_tuple('0.21.3') <= version_tuple(version) < version_tuple('1.3')


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

    if _lacks_missing_go_to_left(version_in):
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

    if _lacks_missing_go_to_left(version_in):
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


def _serialize_tree_state(model, version_in: str) -> dict:
    """
    Serialize the low-level tree of a fitted decision tree estimator.

    Parameters
    ----------
    model : DecisionTreeRegressor or DecisionTreeClassifier
        A fitted scikit-learn decision tree instance.
    version_in : str
        The sklearn version used to train the model.

    Returns
    -------
    dict
        Dictionary with max_depth, node_count, values, nodes and dtypes.
    """

    state = model.tree_.__getstate__()

    return {
        'max_depth': int(state['max_depth']),
        'node_count': int(state['node_count']),
        'values': state['values'].tolist(),
        'nodes': [list(n) for n in _get_extended_nodes(state['nodes'].tolist(), version_in)],
        'dtypes': _build_dtype_dict(state['nodes'].dtype, version_in)
    }


def _load_tree_arrays(data: dict, version_out: str) -> tuple:
    """
    Rebuild the nodes and values arrays of a serialized tree.

    Parameters
    ----------
    data : dict
        Dictionary produced by a decision tree serializer.
    version_out : str
        The sklearn version of the target environment.

    Returns
    -------
    tuple
        A tuple of (np.ndarray, np.ndarray) with the nodes and the values.
    """

    serialized = data['serialized_tree']

    tree_dtype, num_elements = _build_tree_dtype(serialized['dtypes'], version_out)

    serialized['nodes'] = [tuple(n[:num_elements]) for n in serialized['nodes']]
    nodes_array = np.array(serialized['nodes'], dtype=tree_dtype)
    values_array = np.array(serialized['values'])

    return nodes_array, values_array


def _build_tree(data: dict, nodes_array: np.ndarray, values_array: np.ndarray, n_classes: np.ndarray) -> Tree:
    """
    Reconstruct the low-level Tree object of a decision tree estimator.

    Parameters
    ----------
    data : dict
        Dictionary produced by a decision tree serializer.
    nodes_array : np.ndarray
        Nodes of the tree, with the dtype of the target sklearn version.
    values_array : np.ndarray
        Values of the tree.
    n_classes : np.ndarray
        Number of classes per output.

    Returns
    -------
    Tree
        The reconstructed low-level tree.
    """

    serialized = data['serialized_tree']
    n_features = (data['n_features'] or data['n_features_in'])

    tree_obj = Tree(n_features, n_classes, data['n_outputs'])
    tree_obj.__setstate__({
        'max_depth': serialized['max_depth'],
        'node_count': serialized['node_count'],
        'nodes': nodes_array,
        'values': values_array
    })

    return tree_obj
