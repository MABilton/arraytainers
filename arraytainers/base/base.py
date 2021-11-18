from np.lib.mixins import NDArrayOperatorsMixin
from jaxlib.xla_extension import DeviceArray

# Import methods defined in other files:
import .get, .iters, .functions, .set

class Arraytainer(NDArrayOperatorsMixin, .get.Mixin, 
                  .iters.Mixin, .functions.Mixin, .set.Mixin):

  ARRAY_TYPES = (np.ndarray, DeviceArray)

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

