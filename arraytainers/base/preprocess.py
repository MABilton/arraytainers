
class Mixin:
  
  def _containerise_contents(self, convert_to_arrays):
    for key, val in self.items():
      if isinstance(val, (list, tuple, dict)):
        self.contents[key] = self.__class__(val, convert_to_arrays=convert_to_arrays)

      # If we have a Numpytainer in a Jaxtainer or vice versa:
      elif self.is_container(val) and not isinstance(val, self.__class__):
        self.contents[key] = self.__class__(val.unpacked)

  def _convert_contents_to_arrays(self, greedy_array_conversion):
    converted_contents, _ = self._convert_contents_to_arrays_recursion(self.contents, greedy_array_conversion, initial_call=True)
    self.contents = converted_contents

  def _convert_contents_to_arrays_recursion(self, contents, greedy, initial_call=False):

    try:
      keys = contents.keys()
    except AttributeError:
      keys = range(len(contents))

    can_convert = {}
    print_flag=False

    for key in keys:
      
      val = contents[key]

      # Contents of a container should already be converted:
      if self.is_container(val):
        can_convert[key] = False
      
      # Can't convert dictionary to array, but may be able to convert values in dictionary to array(s):
      elif isinstance(val, dict):
        can_convert[key] = False
        contents[key], _ = self._convert_contents_to_arrays_recursion(val, greedy)
      
      elif isinstance(val, (list, tuple)):
        if not greedy:
          # Can convert list to array only if its contents can be converted to arrays:
          contents[key], can_convert[key] = self._convert_contents_to_arrays_recursion(val, greedy)
        else:
          # Immediately convert list/tuple to array if all it's elements are convertible:
          _, can_convert_list = self._convert_contents_to_arrays_recursion(val, greedy)
          if can_convert_list:
            contents[key] = self.array(val)
          can_convert[key] = False # Can't convert list now that it's an array
      
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
    # If everything is convertable after the first call, we've been given a list or dict of
    # directly convertible types - let's directly convert these to arrays:
    elif initial_call:
      try:
        contents = {key:self.array(val) for key, val in contents.items()}  
      except AttributeError:
        contents = [self.array(val) for val in contents]

      # # If we have only one convertable val, convert that val:
      # if len(contents)==1:
      #   first_key = tuple(keys)[0]
      #   contents[first_key] = self.array(contents[first_key])

    return (contents, can_convert_all_vals)