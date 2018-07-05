import warnings
from dltk.core.units import vanilla_residual_unit_3d as _res_unit


def vanilla_residual_unit_3d(*args, **kwargs):
    warnings.warn(
        'This function moved to dltk.core.units.vanilla_residual_unit_3d',
        DeprecationWarning)
    return _res_unit(*args, **kwargs)
