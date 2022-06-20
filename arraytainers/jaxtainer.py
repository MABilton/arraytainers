from .arraytainer import Arraytainer
import copy
import numpy as np
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from inspect import getmembers, ismodule

_JNP_SUBMODULES = (jnp, *[module for _, module in getmembers(jnp, ismodule)])

@register_pytree_node_class
class Jaxtainer(Arraytainer):

    def __init__(self, contents, array_conversions=True, dtype=None):
        super().__init__(contents, array_conversions=array_conversions, dtype=dtype)

    #
    #   Overridden Methods
    #
        
    @classmethod
    def _deepcopy_contents(cls, contents):
        contents_copy = copy.copy(contents)
        items = contents.items() if isinstance(contents, dict) else enumerate(contents)
        for key, val in items:
            # Jax arrays converted to Numpy arrays if copied (Jax arrays are immutable):
            if cls.is_array(val):
                contents_copy[key] = val
            elif isinstance(val, Arraytainer):
                contents_copy[key] = cls._deepcopy_contents(val.contents)
            elif isinstance(val, (list, dict)):
                contents_copy[key] = cls._deepcopy_contents(val)
            else:
                contents_copy[key] = copy.deepcopy(val)
        return contents_copy

    def __deepcopy__(self, memo):
        deepcopied_contents = self._deepcopy_contents(self._contents)
        return self.__class__(deepcopied_contents)

    def _set_array_values(self, key, mask, value):
        self._contents[key] = self._contents[key].at[mask].set(value)

    @staticmethod
    def create_array(val, dtype=None):
        return jnp.array(val, dtype=dtype)

    @staticmethod
    def is_array(val):
        return isinstance(val, (np.ndarray, jnp.ndarray))

    #
    #   Numpy Function Handling Methods (Overrides Arraytainer methods)
    #

    def _manage_func_call(self, func, types, *args, **kwargs):
        arraytainer_list = self._list_arraytainers_in_args(args) + self._list_arraytainers_in_args(kwargs)
        largest_arraytainer = max(arraytainer_list, key=len)
        self._check_arg_compatability(arraytainer_list, largest_arraytainer)
        shared_keyset = self._get_shared_keyset(arraytainer_list)
        func_return = copy.deepcopy(largest_arraytainer.contents)
        for key in shared_keyset:
            args_i = self._prepare_func_args(args, key)
            kwargs_i = self._prepare_func_args(kwargs, key)
            arraytainer_list_i = self._list_arraytainers_in_args(args_i) + self._list_arraytainers_in_args(kwargs_i)
            # Need to call Numpy method for recursion on Arraytainer arguments:
            method = self._find_jnp_method(func) if not arraytainer_list_i else func
            func_return[key] = method(*args_i, **kwargs_i)
        return self.__class__(func_return)
        
    def _prepare_func_args(self, args, key):
        prepped_args = super()._prepare_func_args(args, key)
        # Jax methods don't use 'out' keyword in kwargs:
        if isinstance(prepped_args, dict):
            prepped_args.pop('out', None)
        return prepped_args

    def _find_jnp_method(self, func):
        method_name = str(func.__name__)
        for i, submodule in enumerate(_JNP_SUBMODULES):
            if hasattr(submodule, method_name):
                found_method = getattr(submodule, method_name)
                break
            elif i == len(_JNP_SUBMODULES)-1:
                raise NotImplementedError(f'The {method_name} method is not implemented in jax.numpy.')
        return found_method

    #
    #   Jax Tree Methods
    #

    def tree_flatten(self):
        return jax.tree_util.tree_flatten(self.unpack())

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(jax.tree_util.tree_unflatten(aux_data, children), array_conversions=False)