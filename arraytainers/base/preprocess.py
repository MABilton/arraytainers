
class Mixin:
  
  def _containerise_contents(self):
    for key, val in self.items():
      
      if isinstance(val, (list, dict)):
        self.contents[key] = self.__class__(val)

      # If we have a Numpytainer in a Jaxtainer or vice versa:
      elif self.is_container(val) and not isinstance(val, self.__class__):
        self.contents[key] = self.__class__(val.unpacked)

  def _convert_contents_to_arrays(self):
    converted_contents, _ = self._convert_contents_to_arrays_recursion(self.contents)
    self.contents = converted_contents

  def _convert_contents_to_arrays_recursion(self, contents):

    try:
      keys = contents.keys()
    except AttributeError:
      keys = range(len(contents))
    keys = tuple(keys)

    can_convert = {}

    for key in keys:
      
      val = contents[key]

      # Can't convert dictionary to array, but may be able to convert values in dictionary to array(s):
      if isinstance(val, dict):
        can_convert[key] = False
        contents[key], _ = self._convert_contents_to_arrays_recursion(val)
      
      # Can convert list to array only if its contents can be converted to arrays:
      elif isinstance(val, (list, tuple)):
        contents[key], can_convert[key] = self._convert_contents_to_arrays_recursion(list(val))
      
      # If val is a Numpy array in a Numpytainer or Jax array in a Jaxtainer:
      elif isinstance(val, self.array_type):
        can_convert[key] = False
      
      # If val is a Jax array in a Numpytainer or a Numpy array in a Jaxtainer:
      elif self.is_array(val):
        contents[key] = self.array(val)
        can_convert[key] = False
      
      # If val is an int, float, string, etc:
      else:
        can_convert[key] = True

    can_convert_all_vals = all(can_convert.values())

    # Convert individual vals if we can't convert all vals at this level: 
    if not can_convert_all_vals:
      for key, covert_flag in can_convert.items():
        contents[key] = self.array(contents[key]) if covert_flag else contents[key]
    # If we have only one convertable val, convert that val:
    elif len(keys)==1:
      first_key = keys[0]
      contents[first_key] = self.array(contents[first_key])

    return (contents, can_convert_all_vals)