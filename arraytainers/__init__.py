from warnings import warn

from .arraytainer import Arraytainer
try:
    from .jaxtainer import Jaxtainer
except ImportError:
    warn('Failed to import Jaxtainers; please visit '  
         'https://github.com/google/jax#installation for ' 
         'details on installing Jax.')

# Clean-up namespace of module:
del warn