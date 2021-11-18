from more_itertools.more import always_iterable

class Mixin:
    
    def append(self, new_val):
        try:
            self.contents.append(new_val)
        except AttributeError:
            error_msg = """Unable to append values to dictionary-like container; 
                        use 'my_container[new_key] = new_value' instead."""
            raise AttributeError(error_msg)

    def __setitem__(self, key, new_value):
        key_type = type(key)
        if issubclass(key_type, Arraytainer):
            self._set_with_container(key, new_value)
        elif key_type in self.ARRAY_TYPES:
            self._set_with_array(key, new_value)
        elif all(isinstance(val, slice) for val in always_iterable(key)):
            self._set_with_slices(key, new_value)
        # If we're given a tuple of integers:
        elif isinstance(key, tuple) and all(isinstance(val, int) for val in always_iterable(key)):
            self._set_with_slices(key, new_value)
        else:
            self._set_with_hash(key, new_value)    

    def _set_with_container(self, container, new_value):
        value_is_container = issubclass(type(new_value), Arraytainer)
        for container_key in self.keys():
            idx = container[container_key]
            value_i = new_value[container_key] if value_is_container else new_value
            self._set_array_item(container_key, idx, value_i)

    def _set_with_array(self, array_key, new_value):
        value_is_container = issubclass(type(new_value), Arraytainer)
        for container_key in self.keys():
            value_i = new_value[container_key] if value_is_container else new_value
            self._set_array_item(container_key, array_key, value_i)

    def _set_with_slices(self, slices, new_value):
        value_is_container = issubclass(type(new_value), Arraytainer)
        for container_key in self.keys():
            value_i = new_value[container_key] if value_is_container else new_value
            self._set_array_item(container_key, slices, value_i)

    def _set_array_item(self, key, idx, value_i):
        error_msg = '''Cannot use array to set value with base Arraytainer class;
                    use a Numpytainer or Jaxtainer instead.'''
        return KeyError(error_msg)

    def _set_with_hash(self, key, new_value):
        try:
            self.contents[key] = new_value
        except IndexError:
            if self._type is list:
                error_msg = "Unable to assign items to a list-like container; use the append method instead."
                raise TypeError(error_msg)
            else:
                error_msg = f"Provided key {key} is invalid."
                raise IndexError(error_msg)