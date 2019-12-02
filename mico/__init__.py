"""
MICO: Mutual Information and Conic Optimization for feature selection.

Github repo : https://github.com/jupiters1117/mico
Author      : KuoLing Huang <jupiters1117@gmail.com>
License     : BSD 3 clause


Note
----
MICO is heavily inspired from MIFS by Daniel Homola:

Github repo : https://github.com/danielhomola/mifs
Author      : Daniel Homola <dani.homola@gmail.com>
License     : BSD 3 clause
"""
from .mico import MutualInformationForwardSelection,  MutualInformationBackwardElimination, MutualInformationConicOptimization
from .version import _check_module_dependencies, __version__

_check_module_dependencies()

# Boolean controlling whether the joblib caches should be
# flushed if the version of certain modules changes (eg nibabel, as it
# does not respect the backward compatibility in some of its internal
# structures
CHECK_CACHE_VERSION = True

# list all available submodules and version
__all__ = \
[
    'MutualInformationForwardSelection',
    ' MutualInformationBackwardElimination',
    'MutualInformationConicOptimization',
    '__version__'
]
