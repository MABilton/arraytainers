from .arraytainer import Arraytainer
import copy
import numpy as np
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
class Jaxtainer(Arraytainer):

    _arrays = (np.ndarray, jnp.DeviceArray)

    def __init__(self, contents, convert_arrays=True, greedy=False, nested=True):
        super().__init__(contents, convert_arrays, greedy, nested)

    #
    #   Overridden Methods
    #

    def __deepcopy__(self, memo):
        deepcopied_contents = self._deepcopy_contents(self.unpack())
        return self.__class__(deepcopied_contents)

    def _deepcopy_contents(self, contents):
        
        copied_contents = contents.copy()
        contents_iter = contents.items() if isinstance(contents, dict) else enumerate(contents)
        
        for key, val in contents_iter:
            # Jax arrays converted to Numpy arrays if copied (Jax arrays are immutable):
            if isinstance(val, self._arrays):
                copied_contents[key] = self._convert_to_array(val)
            elif isinstance(val, (list, dict, tuple)):
                copied_contents[key] = self._deepcopy_contents(val)
            else:
                copied_contents[key] = copy.deepcopy(val)

        return copied_contents

    def _set_array_values(self, key, idx, value):
        self._contents[key] = self.contents[key].at[idx].set(value)

    @staticmethod
    def _convert_to_array(val):
        return jnp.array(val)

    @staticmethod
    def _extract_array_vals(vector, elem_idx, shape):
        num_elem = jnp.prod(shape)
        array_vals = vector[elem_idx:elem_idx+num_elem]
        elem_idx += num_elem
        return array_vals, elem_idx

    #
    #   Numpy Function Handling Methods (Overrides Arraytainer methods)
    #

    def _manage_func_call(self, func, types, *args, **kwargs):

        func_return = {}

        arraytainer_list = self._list_arraytainers_in_args(args) + self._list_arraytainers_in_args(kwargs)
        largest_arraytainer = self._find_largest_arraytainer(arraytainer_list)
        self._check_arraytainer_arg_compatability(arraytainer_list, largest_arraytainer)
        shared_keyset = self._get_shared_keyset(arraytainer_list)

        func_return = copy.deepcopy(largest_arraytainer.contents)
        for key in shared_keyset:
            
            args_i = self._prepare_func_args(args, key)
            kwargs_i = self._prepare_func_args(kwargs, key)
            arraytainer_list_i = self._list_arraytainers_in_args(args_i) + self._list_arraytainers_in_args(kwargs_i)

            # Need to call Numpy method for recursion on Arraytainer arguments:
            method = self._find_jnp_method(func) if not arraytainer_list_i else func
            func_return[key] = method(*args_i, **kwargs_i)

        return self.__class__(func_return, greedy=True)
        
    def _prepare_func_args(self, args, key):
        prepped_args = super()._prepare_func_args(args, key)
        # Jax methods don't use 'out' keyword in kwargs:
        if isinstance(prepped_args, dict):
            prepped_args.pop('out', None)
        return prepped_args

    _jnp_submodules_to_search = ('', 'linalg', 'fft')
    def _find_jnp_method(self, func):

        method_name = str(func.__name__)

        for i, submod_name in enumerate(self._jnp_submodules_to_search):

            if submod_name != "":
                submodule = getattr(jnp, submod_name)
            else:
                submodule = jnp

            if hasattr(submodule, method_name):
                found_method = getattr(submodule, method_name)
                break
            elif i == len(self._jnp_submodules_to_search)-1:
                raise AttributeError(f'The {method_name} method is not implemented in jax.numpy.')

        return found_method

    #
    #   Jax Tree Methods
    #

    def tree_flatten(self):
        return jax.tree_util.tree_flatten(self.unpacked)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        try:
            unflattened = \
                cls(jax.tree_util.tree_unflatten(aux_data, children), convert_arrays=True, nested=True)
        except TypeError:
            unflattened = \
                cls(jax.tree_util.tree_unflatten(aux_data, children), convert_arrays=False, nested=False)
        return unflattened