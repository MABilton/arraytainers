from warnings import warn

from .numpytainer import Numpytainer
try:
    from .jaxtainer import Jaxtainer
except ModuleNotFoundError:
    warning_msg = '''Failed to import Jaxtainers; make sure Jax 
                     and Jaxlib are correctly installed on your machine.'''
    warn(warning_msg)