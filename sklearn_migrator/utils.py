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
