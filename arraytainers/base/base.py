from copy import deepcopy
import functools

import numpy as np
from numpy import ndarray
from numpy.lib.mixins import NDArrayOperatorsMixin

from . import getset, func_handlers, preprocess

def get_supported_arrays():
  
  supported_arrays = [ndarray]
  
  try:
      from jax.numpy import DeviceArray
      supported_arrays.append(DeviceArray)
  except ModuleNotFoundError:
      pass
  
  return tuple(supported_arrays)

class Arraytainer(NDArrayOperatorsMixin, getset.GetterMixin, 
                  getset.SetterMixin, func_handlers.Mixin, preprocess.Mixin):

    supported_arrays = get_supported_arrays()

    def __init__(self, contents):
        contents = list(contents) if isinstance(contents, tuple) else contents
        self.contents = deepcopy(contents)
        self._type = dict if hasattr(self.contents, 'keys') else list
        self.check_keys()

    # Generic methods:       
    def __len__(self):
        return len(self.contents)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.unpacked)})"

    def copy(self):
        return self.__class__(deepcopy(self.contents))

    # Array methods/proprties:
    def array(self, x):
        if self.is_container(x):
            return x
        else:
            return self.array_class(x)

    @property
    def T(self):
        return np.transpose(self)

    # Key/Item methods:
    def keys(self):
        if self._type is dict:
            keys = [key for key in self.contents.keys()]
        else:
            keys = tuple(i for i in range(len(self.contents)))
        return keys

    def values(self, unpacked=False):
        contents = self.unpacked if unpacked else self
        return tuple(contents[key] for key in self.keys())

    def items(self, unpacked=False):
        contents = self.unpacked if unpacked else self
        return tuple((key, contents[key]) for key in self.keys())

    # Calculation methods:
    def all(self):
        for key in self.keys():
            if self.contents[key].all():
                continue
            else:
                return False
        return True

    def any(self):
        for key in self.keys():
            if self.contents[key].any():
                return True
        return False

    def sum(self, elements=True):
        return self.sum_elements() if elements else self.sum_arrays()

    def sum_elements(self):
        if len(self.keys()) > 0:
            sum_results = functools.reduce(lambda x,y: x+y, [val for val in self.values()])
        else:
            sum_results = 0
        return sum_results

    def sum_arrays(self):
        to_sum = self.list_arrays()
        return sum(to_sum)

    # Type-checking methods:
    def is_array(self, val):
        return isinstance(val, self.supported_arrays)

    @staticmethod
    def is_container(val):
        return issubclass(type(val), Arraytainer)

    # Shape methods:
    @property
    def shape(self):
        shapes = [self[key].shape for key in self.keys()]
        if self._type is dict:
            shapes = dict(zip(self.keys(), shapes))
        return shapes

    @property
    def shape_container(self):
        shapes = self.shape
        return self.__class__(shapes)

    # Convert arraytainer back to 'normal' dicts/lists:
    @property
    def unpacked(self):
        output = [val.unpacked if self.is_container(val) else val
                  for val in self.values()]
        if self._type is dict:
            output = dict(zip(self.keys(), output))
        return output
    
    # Returns list of all arrays in arraytainer:
    def list_arrays(self):
        unpacked = self.unpacked
        return flatten_contents(unpacked)

# Helper functions
def flatten_contents(contents, array_list=None):

    array_list = [] if array_list is None else array_list
    keys = range(len(contents)) if isinstance(contents, list) else contents.keys()

    for key in keys:
        if isinstance(contents[key], (dict, list)):
            array_list = flatten_contents(contents[key], array_list)
        else:
            array_list.append(contents[key])
    
    return array_list