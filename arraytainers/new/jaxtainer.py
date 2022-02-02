from .arraytainer import Arraytainer

import jax
from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
class Jaxtainer(Arraytainer):

    _arrays = (np.ndarray, jnp.ndarray)
    _jnp_submodules_to_search = ('', 'linalg', 'fft')

    def __init__(self, contents, convert_arrays=True, greedy=False, floats_only=False, nested=True):
        self.floats_only = floats_only
        super().__init__(contents, convert_arrays, greedy, nested)

    #
    #   Array Methods (Replaces Arraytainer methods)
    #

    def _set_array_val(self, key, idx, value):
        self._contents[key] = self.contents[key].at[idx].set(value)

    def _convert_to_array(self, val):
        if self.floats_only:
            array = jnp.array(val).astype(float)
        else:
            array = jnp.array(val)
        return array

    #
    #   Numpy Function Handling Methods (Replaces Arraytainer methods)
    #

    def _manage_function_call(self, func, types, *args, **kwargs):

        func_return = {}

        arraytainer_list = self._find_arraytainers_in_args(args, kwargs)
        largest_arraytainer = self._find_largest_arraytainer(arraytainer_list)
        self._check_arraytainer_arg_compatability(arraytainer_list, largest_arraytainer)

        for key in largest_arraytainer.keys():
            
            args_i = self._prepare_args(args, key)
            kwargs_i = self._prepare_kwargs(kwargs, key)
            arraytainer_list_i = self._find_arraytainers_in_args(args_i, kwargs_i)

            # Need to call Numpy method for recursion on Arraytainer arguments:
            method = self._find_jnp_method(func) if not arraytainer_list_i else func
            func_return[key] = method(*args_i, **kwargs_i)

        if self._type is list:
            func_return = list(func_return.values())

        return self.__class__(func_return)
        
    def _prepare_kwargs(self, kwargs, key):
        kwargs = super()._prepare_kwargs(kwargs, key)
        # Jax methods don't use 'out' keyword:
        kwargs.pop('out', None)
        return kwargs

    def _find_containers(self, args, kwargs):
        containers_in_args = [self.is_container(arg_i) for arg_i in args]
        containers_in_kwargs = [self.is_container(arg_i) for arg_i in kwargs.values()]
        includes_containers = any(containers_in_args) or any(containers_in_kwargs)
        return includes_containers

    def _find_jnp_method(func):

        method_name = str(func.__name__)

        for i, submod_name in enumerate(self._jnp_submodules_to_search):

            if submod_name != "":
                submodule = getattr(jnp, submod_name)
            else:
                submodule = jnp

            if hasattr(submodule, method_name):
                found_method = getattr(submodule, method_name)
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