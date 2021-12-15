import functools

import numpy as np
from numpy import ndarray
from numpy.lib.mixins import NDArrayOperatorsMixin
from copy import deepcopy

from . import getset, func_handlers, preprocess, arrays

def get_supported_arrays():
  
  supported_arrays = [ndarray]
  
  try:
      from jax.numpy import DeviceArray
      supported_arrays.append(DeviceArray)
  except ModuleNotFoundError:
      pass
  
  return tuple(supported_arrays)

class Arraytainer(NDArrayOperatorsMixin, getset.GetterMixin, arrays.Mixin, 
                  getset.SetterMixin, func_handlers.Mixin, preprocess.Mixin):

    supported_arrays = get_supported_arrays()

    def __init__(self, contents):

        if self.is_array(contents):
            contents = [contents]
        elif self.is_container(contents):
            contents = contents.unpacked
        elif not hasattr(contents, '__len__'):
            contents = [contents]
        elif isinstance(contents, tuple):
            contents = list(contents)

        # Need to be careful when copying Jax arrays - this actually returns a Numpy array:
        self.contents = deepcopy(contents)
        self._type = dict if hasattr(self.contents, 'keys') else list
        self.check_keys()

    @classmethod
    def from_vector(cls, vector, shapes, order='C'):

        try:
            vector = vector.flatten(order=order).tolist()
        except AttributeError:
            error_msg = ('Only arrays which can be converted to lists are', 
                         'allowed to be passed to the from_vector constructor.')
            raise ValueError(' '.join(error_msg))

        if isinstance(shapes, (list, dict, tuple)):
            shapes = cls(shapes, greedy_array_conversion=True)
        elif not cls.is_container(shapes):
            raise ValueError('Shapes input must be either a list, dict, or arraytainer.')
        
        new_contents = from_vector_recursion(vector, shapes, order)

        return cls(new_contents)

    # Generic methods:       
    def __len__(self):
        return len(self.contents)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.unpacked)})"

    def copy(self):
        return self.__class__(deepcopy(self.unpacked))

    # Key/Item methods:
    def keys(self):
        if self._type is dict:
            keys = (key for key in self.contents.keys())
        else:
            keys = (i for i in range(len(self.contents)))
        return keys

    def values(self, unpacked=False):
        contents = self.unpacked if unpacked else self
        return (contents[key] for key in self.keys())

    def items(self, unpacked=False):
        contents = self.unpacked if unpacked else self
        return ((key, contents[key]) for key in self.keys())

    def __iter__(self):
        return self.values()

    # Type-checking methods:
    @classmethod
    def is_array(cls, val):
        return isinstance(val, cls.supported_arrays)
        
    @staticmethod
    def is_container(val):
        return isinstance(val, Arraytainer)

    # Convert arraytainer back to 'normal' dicts/lists:
    def unpack(self):
        output = [val.unpacked if self.is_container(val) else val
                    for val in self.values()]
        if self._type is dict:
            output = dict(zip(self.keys(), output))
        return output

    @property
    def unpacked(self):
        return self.unpack()

    # Returns list of all arrays in arraytainer:
    def list_arrays(self):
        unpacked = self.unpacked
        return flatten_contents(unpacked)

# Helper functions

def from_vector_recursion(vector, shapes, order, output=None):
    
    if output is None:
        output = shapes.copy().unpacked

    for key, shape in shapes.items():
        if Arraytainer.is_array(shape):
            num_vals = np.prod(shape)
            vals_i = np.array([vector.pop(0) for _ in range(num_vals)])
            output[key] = vals_i.reshape(shape, order=order)
        else:
            output[key] = from_vector_recursion(vector, shape, order, output=output[key])
    
    return output

def flatten_contents(contents, array_list=None):

    array_list = [] if array_list is None else array_list
    keys = range(len(contents)) if isinstance(contents, list) else contents.keys()

    for key in keys:
        if isinstance(contents[key], (dict, list)):
            array_list = flatten_contents(contents[key], array_list)
        else:
            array_list.append(contents[key])
    
    return array_list