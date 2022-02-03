from .arraytainer import Arraytainer
import numpy as np
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
class Jaxtainer(Arraytainer):

    _arrays = (np.ndarray, jnp.DeviceArray)
    _jnp_submodules_to_search = ('', 'linalg', 'fft')

    def __init__(self, contents, convert_arrays=True, greedy=False, floats_only=False, nested=True):
        self.floats_only = floats_only
        super().__init__(contents, convert_arrays, greedy, nested)

    #
    #   Array Methods (Overrides Arraytainer methods)
    #

    def _deepcopy(self, contents):
        copied_contents = contents.copy()
        contents_iter = contents.items() if isinstance(contents, dict) else enumerate(contents)
        for key, val in contents_iter:
            if isinstance(val, self._arrays):
                copied_contents[key] = val
            elif isinstance(val, (list, dict, tuple)):
                copied_contents[key] = self._deepcopy(val)
            else:
                copied_contents[key] = val.copy() 
        return copied_contents

    def _set_array_val(self, key, idx, value):
        self._contents[key] = self.contents[key].at[idx].set(value)

    def _convert_to_array(self, val):
        array = jnp.array(val)
        if self.floats_only:
            array = array.astype(float)   
        return array

    @staticmethod
    def _extract_array_vals(vector, shape):
        num_elem = jnp.prod(shape)
        return jnp.array([vector.pop(0) for _ in range(num_elem)])

    #
    #   Numpy Function Handling Methods (Overrides Arraytainer methods)
    #

    def _manage_func_call(self, func, types, *args, **kwargs):

        func_return = {}

        arraytainer_list = self._list_arraytainers_in_args(args) + self._list_arraytainers_in_args(kwargs)
        largest_arraytainer = self._find_largest_arraytainer(arraytainer_list)
        self._check_arraytainer_arg_compatability(arraytainer_list, largest_arraytainer)

        for key in largest_arraytainer.keys():
            
            args_i = self._prepare_func_args(args, key)
            kwargs_i = self._prepare_func_args(kwargs, key)
            arraytainer_list_i = self._list_arraytainers_in_args(args_i) + self._list_arraytainers_in_args(kwargs_i)

            # Need to call Numpy method for recursion on Arraytainer arguments:
            method = self._find_jnp_method(func) if not arraytainer_list_i else func
            func_return[key] = method(*args_i, **kwargs_i)

        if self._type is list:
            func_return = list(func_return.values())

        return self.__class__(func_return, greedy=True)
        
    def _prepare_func_args(self, args, key):
        prepped_args = super()._prepare_func_args(args, key)
        # Jax methods don't use 'out' keyword in kwargs:
        if isinstance(prepped_args, dict):
            prepped_args.pop('out', None)
        return prepped_args

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