import warnings
import numpy as np


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

    version_split = version.split(".")

    if len(version_split) == 1:
        new_version = (int(version_split[0]), 0, 0)
    elif len(version_split) == 2:
        new_version = (int(version_split[0]), int(version_split[1]), 0)
    elif len(version_split) == 3:
        new_version = (int(version_split[0]), int(version_split[1]), int(version_split[2]))
    else:
        new_version = (0, 0, 0)

    return new_version


def json_convert(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: json_convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_convert(i) for i in obj]
    return obj


def get_attr_or_none(obj, name: str):
    """
    Return an attribute of an object, or None if it cannot be read.

    Parameters
    ----------
    obj : object
        Object to read the attribute from.
    name : str
        Name of the attribute.

    Returns
    -------
    object
        The attribute value, or None if it is not available.
    """

    try:
        return getattr(obj, name)
    except Exception:
        return None


def set_attr_safely(obj, name: str, value, warn: bool = False) -> None:
    """
    Set an attribute on an object, ignoring attributes that are not settable
    in the current sklearn version.

    Parameters
    ----------
    obj : object
        Object to set the attribute on.
    name : str
        Name of the attribute.
    value : object
        Value to set.
    warn : bool
        If True, emit a warning for unexpected errors instead of ignoring them.
    """

    try:
        setattr(obj, name, value)
    except (KeyError, AttributeError):
        pass
    except Exception as e:
        if warn:
            warnings.warn(f"Could not set field '{name}': {type(e).__name__}: {e}")


def collect_other_params(model, all_features: list, default_values: dict) -> dict:
    """
    Read a list of fields from a fitted model, falling back to default values
    for the fields that do not exist in the sklearn version of the model.

    Parameters
    ----------
    model : object
        A fitted scikit-learn estimator.
    all_features : list
        Names of the fields to read from the model.
    default_values : dict
        Values to use for the fields that are not present in the model.

    Returns
    -------
    dict
        Dictionary with one entry per field in all_features.
    """

    model_dict = model.__dict__

    return {
        af: model_dict[af] if af in model_dict else default_values[af]
        for af in all_features
    }


def restore_other_params(model, all_features: list, data: dict) -> None:
    """
    Set the fields stored in data['other_params'] on a reconstructed model.

    Parameters
    ----------
    model : object
        The scikit-learn estimator being reconstructed.
    all_features : list
        Names of the fields to set on the model.
    data : dict
        Serialized dictionary containing the 'other_params' entry.
    """

    for af in all_features:
        try:
            model.__dict__[af] = data['other_params'][af]
        except KeyError:
            pass  # field not present in this sklearn version
        except AttributeError:
            pass  # attribute not settable in this sklearn version
        except Exception as e:
            warnings.warn(
                f"Could not set field '{af}' on {type(model).__name__}: "
                f"{type(e).__name__}: {e}. Field will be skipped.",
                UserWarning,
            )


def filter_init_params(model_class, params: dict) -> dict:
    """
    Keep only the constructor parameters accepted by the current sklearn version.

    Parameters
    ----------
    model_class : type
        The scikit-learn estimator class to instantiate.
    params : dict
        Parameters obtained with get_params() in the source environment.

    Returns
    -------
    dict
        The subset of params accepted by model_class in this environment.
    """

    return {
        param: params[param]
        for param in model_class().get_params()
        if param in params
    }
