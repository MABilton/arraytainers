# Try import jaxtainers - fails if jax not installed:
from warnings import warn

from .arraytainer import Arraytainer

__version__ = "1.0.0"
__all__ = ["Arraytainer"]


try:
    from .jaxtainer import Jaxtainer

    __all__.append("Jaxtainer")
except ImportError:
    jax_url = "https://github.com/google/jax#installation"
    warn(
        f"Could not import jaxtainers since jax is not installed; please visit {jax_url} for details on installing jax."
    )
# Clean-up namespace of module:
del warn
