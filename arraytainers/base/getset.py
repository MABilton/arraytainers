
from more_itertools.more import always_iterable

def is_slice(key):
    return all(isinstance(val, slice) for val in always_iterable(key))

def is_tuple_of_ints(key):
    return isinstance(key, tuple) and all(isinstance(val, int) for val in always_iterable(key))

class GetterMixin:
    
    def check_keys(self):
        invalid_keys = [str(key) for key in self.keys() if is_tuple_of_ints(key)]
        if invalid_keys:
            raise KeyError(f'Cannot use {", ".join(invalid_keys)}, tuple(s) of ints, as key(s) in an Arraytainer.')

    def __getitem__(self, key):
        if self.is_container(key):
            item = self._index_with_container(key)
        elif self.is_array(key):
            item = self._index_with_array(key)
        elif is_slice(key) or is_tuple_of_ints(key):
            item = self._index_with_slices(key)
        else:
            item = self._index_with_hash(key)
        return item

    def _index_with_container(self, key_container):
        item = self.copy()
        for container_key in self.keys():
            array_key = key_container[container_key]  
            item[container_key] = self.contents[container_key][self.array(array_key)]
        return item

    def _index_with_array(self, array_key):
        item = self.copy()
        for container_key in self.keys():
            item[container_key] = self.contents[container_key][self.array(array_key)]
        return item

    def _index_with_slices(self, slices):
        item = self.copy()
        for container_key in self.keys():
            item[container_key] = self.contents[container_key][slices]
        return item

    def _index_with_hash(self, key):
        try:
            item = self.contents[key]
        except KeyError:
            error_msg = (f'{key} is not a key in the {self.__class__.__name__};',
                         f'valid keys for this {self.__class__.__name__} are {tuple(self.keys())}.')
            raise KeyError(' '.join(error_msg))
        return item

class SetterMixin:
    
    def append(self, new_val):
        try:
            self.contents.append(new_val)
        except AttributeError:
            error_msg = ('Unable to append values to dictionary-like container;',
                         "use the syntax 'Arraytainer[new_key] = new_value' instead.")
            raise AttributeError(' '.join(error_msg))

    def __setitem__(self, key, new_value):
        if self.is_container(key):
            self._set_with_container(key, new_value)
        elif self.is_array(key):
            self._set_with_array(key, new_value)
        elif is_slice(key) or is_tuple_of_ints(key):
            self._set_with_slices(key, new_value)
        else:
            self._set_with_hash(key, new_value)    

    def _set_with_container(self, container, new_value):
        for container_key in self.keys():
            idx = container[container_key]
            value_i = new_value[container_key] if self.is_container(new_value) else new_value
            self._set_array_item(container_key, idx, value_i)

    def _set_with_array(self, array_key, new_value):
        for container_key in self.keys():
            value_i = new_value[container_key] if self.is_container(new_value) else new_value
            self._set_array_item(container_key, array_key, value_i)

    def _set_with_slices(self, slices, new_value):
        for container_key in self.keys():
            value_i = new_value[container_key] if self.is_container(new_value) else new_value
            self._set_array_item(container_key, slices, value_i)

    def _set_array_item(self, key, idx, value_i):
        error_msg = ('Cannot use array to set value with base Arraytainer class;', 
                     'use a Numpytainer or Jaxtainer instead.')
        return KeyError(' '.join(error_msg))

    def _set_with_hash(self, key, new_value):
        try:
            self.contents[key] = new_value
        except IndexError:
            if self._type is list:
                error_msg = ("Unable to assign items to a list-like container;",
                             "use the append method instead.")
                raise TypeError(' '.join(error_msg))
            else:
                raise IndexError(f"Provided key {key} is invalid.")