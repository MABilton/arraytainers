from numpy import ndarray
from numpy.lib.mixins import NDArrayOperatorsMixin
from . import getters, setters, iters, functions

def get_supported_arrays():
  
  supported_arrays = [ndarray]
  
  try:
      from jax.numpy import DeviceArray
      supported_arrays.append(DeviceArray)
  except ModuleNotFoundError:
      pass
  
  return tuple(supported_arrays)

class Arraytainer(NDArrayOperatorsMixin, getters.Mixin, 
                  iters.Mixin, functions.Mixin, setters.Mixin):

  supported_arrays = get_supported_arrays()

  def __init__(self, contents, containerise_values=True):
    self.contents = contents
    self._type = dict if hasattr(self.contents, 'keys') else list     
    # If nested dictionaries, convert all of these:
    if containerise_values:
      to_covert = [key for key in self.keys() 
                   if hasattr(self[key], 'keys') and not issubclass(type(self[key]), Arraytainer)]
      for key in to_covert:
        self[key] = self.__class__(self[key])
                
  def __len__(self):
    return len(self.contents)

  def __repr__(self):
    return f"{self.__class__.__name__}({repr(self.contents)})"
  
  def copy(self):
    return self.__class__(self.contents.copy())
  
  @staticmethod
  def is_container(input):
    return issubclass(type(input), Arraytainer)