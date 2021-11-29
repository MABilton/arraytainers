import numpy as np
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class, tree_flatten, tree_unflatten
from .base.base import Arraytainer

@register_pytree_node_class
class Jaxtainer(Arraytainer):

    # May want only floats stored for autograd purposes:    
    def __init__(self, contents, convert_to_arrays=True, greedy_array_conversion=False, floats_only=False, containerise_contents=True):

        self.array_class = (lambda x : jnp.array(x).astype(float)) if floats_only else jnp.array
        self.array_type = jnp.DeviceArray

        super().__init__(contents)

        if convert_to_arrays:
          self._convert_contents_to_arrays(greedy_array_conversion)
        if containerise_contents:
          self._containerise_contents(convert_to_arrays)

    # Over-rided method definitions:
    def _manage_function_call(self, func, types, *args, **kwargs):

      output_dict = {}

      self._check_container_compatability(args, kwargs)

      for key in self.keys():
        args_i = self._prepare_args(args, key)
        kwargs_i = self._prepare_kwargs(kwargs, key)
        
        # Check to see if args or kwargs contains a container type:
        includes_containers = self._find_containers(args_i, kwargs_i)

        # If function call does not include containers, we need to remove any 'out' kwargs:
        method = find_method(jnp, func) if not includes_containers else find_method(np, func)
        output_dict[key] = method(*args_i, **kwargs_i)

      if self._type is list:
        output_list = list(output_dict.values())
        output_container = self.__class__(output_list)
      else:
        output_container = self.__class__(output_dict)

      return output_container
    
    # Removes 'out' from kwargs - not use by Jax methods:
    def _prepare_kwargs(self, kwargs, key):
      kwargs = super()._prepare_kwargs(kwargs, key)
      kwargs.pop('out', None)
      return kwargs

    def _find_containers(self, args, kwargs):
      containers_in_args = [self.is_container(arg_i) for arg_i in args]
      containers_in_kwargs = [self.is_container(arg_i) for arg_i in kwargs.values()]
      includes_containers = any(containers_in_args) or any(containers_in_kwargs)
      return includes_containers

    # Functions required by @register_pytree_node_class decorator:
    def tree_flatten(self):
      return tree_flatten(self.unpacked)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
      try:
        unflattened = cls(tree_unflatten(aux_data, children), convert_to_arrays=True, containerise_content=True)
      except TypeError:
        unflattened = cls(tree_unflatten(aux_data, children), convert_to_arrays=False, containerise_content=False)
      return unflattened

    def _set_array_item(self, container_key, idx, value_i):
      self.contents[container_key] = self.contents[container_key].at[idx].set(value_i)

def find_method(module, func, submodule_to_search=('', 'linalg', 'fft')):
    method_name = str(func.__name__)
    modules_to_search = [getattr(module, submodule, module) for submodule in submodule_to_search]
    for i, mod in enumerate(modules_to_search):
      try:
          found_method = getattr(mod, method_name)
          break
      except AttributeError:
          if i == len(submodule_to_search)-1:
            raise AttributeError(f'The {method_name} method is not implemented in {module}.')
    return found_method