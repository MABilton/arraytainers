from warnings import warn

from .base import Arraytainer
from .numpytainer import Numpytainer
try:
    from .jaxtainer import Jaxtainer
except ImportError:
    warning_msg = ('WARNING: Failed to import Jaxtainers; please visit '  
                   'https://github.com/google/jax#installation to ' 
                   'see how to correctly install Jax.')
    warn(warning_msg)

# Clean-up namespace of module:
del warn, numpytainer, jaxtainer