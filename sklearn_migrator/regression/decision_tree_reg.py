import numpy as np
from sklearn.tree._tree import Tree
from sklearn.tree import DecisionTreeRegressor

all_features = [
    'max_leaf_nodes',
    'max_features_',
    'min_weight_fraction_leaf',
    'splitter',
    'class_weight',
    'min_impurity_decrease',
    'min_samples_split',
    'criterion',
    'min_samples_leaf',
    'max_depth',
    'n_outputs_',
    'max_features',
    'min_impurity_split',
    'n_classes_',
    'classes_',
    'presort',
    'ccp_alpha',
    'feature_names_in_',
    'monotonic_cst'
]


def version_tuple(version):

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


def _get_extended_nodes(nodes, version_in):
    if (version_tuple('0.21.3') <= version_tuple(version_in)) and (version_tuple(version_in) < version_tuple('1.3')):
        return [node + (0,) for node in nodes]
    return nodes


def _build_dtype_dict(dtypes, version_in):
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


def _get_metadata(model, version_in):

    dict_metadata = {}

    try:
        dict_metadata['n_features_in'] = model.n_features_in_
    except:
        dict_metadata['n_features_in'] = None
    
    try:
        dict_metadata['n_features'] = model.n_features_
    except:
        dict_metadata['n_features'] = None

    try:
        dict_metadata['n_classes'] = model.n_classes_
    except:
        dict_metadata['n_classes'] = None

    dict_metadata['n_outputs'] = model.n_outputs_

    return dict_metadata


def serialize_decision_tree_reg(model, version_in):
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
        'n_classes_': 1,
        'classes_': None,
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


def _build_tree_dtype(dtypes_dict, version_out):
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


def deserialize_decision_tree_reg(data, version_out):
    version_in = data['version_sklearn_in']
    serialized = data['serialized_tree']
    dtype_dict = serialized['dtypes']

    tree_dtype, num_elements = _build_tree_dtype(dtype_dict, version_out)

    serialized['nodes'] = [tuple(n[:num_elements]) for n in serialized['nodes']]
    nodes_array = np.array(serialized['nodes'], dtype=tree_dtype)
    values_array = np.array(serialized['values'])

    n_classes = np.array([1], dtype=np.intp)  # regression
    n_outputs = data['n_outputs']
    n_features = (data['n_features'] or data['n_features_in'])

    tree_obj = Tree(n_features, n_classes, n_outputs)
    tree_obj.__setstate__({
        'max_depth': serialized['max_depth'],
        'node_count': serialized['node_count'],
        'nodes': nodes_array,
        'values': values_array
    })

    new_tree = DecisionTreeRegressor(max_depth=serialized['max_depth'], random_state=42)
    new_tree.tree_ = tree_obj
    new_tree.n_outputs_ = n_outputs

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
        except:
            pass

    return new_tree