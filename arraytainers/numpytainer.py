import numpy as np
from .base.base import Arraytainer

class Numpytainer(Arraytainer):

  def __init__(self, contents, containerise_contents=True):

      super().__init__(contents)
      
      self.array_type = np.ndarray
      self.array_class = np.array

      self._convert_contents_to_arrays()
      if containerise_contents:
        self._containerise_contents()

  # Over-rided method definitions:
  def _manage_function_call(self, func, types, *args, **kwargs):

    output_dict = {}
    
    # Check containers are compatable in operation:
    self._check_container_compatability(args, kwargs)

    for key in self.keys():
      args_i = self._prepare_args(args, key)
      kwargs_i = self._prepare_kwargs(kwargs, key)
      output_dict[key] = func(*args_i, **kwargs_i)
    
    if self._type is list:
      output_list = list(output_dict.values())
      output_container = self.__class__(output_list)
    else:
        output_container = self.__class__(output_dict)

    return output_container

  def _set_array_item(self, container_key, idx, value_i):
    self.contents[container_key][self.array(idx)] = value_i