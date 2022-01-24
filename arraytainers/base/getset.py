
from numbers import Number

def is_slice(key):
    is_tuple_of_slices = isinstance(key, tuple) and all(isinstance(val, (type(None), slice, int)) for val in key)
    is_lone_slice = isinstance(key, (slice, type(None)))
    return is_tuple_of_slices or is_lone_slice

class GetterMixin:
    
    def __iter__(self):
        return ()

    def check_keys(self):
        slice_keys = [str(key) for key in self.keys() if is_slice(key)]
        if slice_keys:
            raise KeyError(f'Cannot use {", ".join(slice_keys)}, which are interpreted as array slices, as key(s) in an Arraytainer.')
        tuple_keys = [str(key) for key in self.keys() if isinstance(key, tuple)]
        if tuple_keys:
            raise KeyError(f'Cannot use the tuples {", ".join(tuple_keys)} as key(s) in an Arraytainer, since tuples are used to extract multiple elements.')

    def filter(self, to_keep, *key_iterable):
        if key_iterable:
            key_iterable = list(key_iterable)
            key_i = key_iterable.pop(0)
            filtered = self.copy()
            filtered[key_i] = filtered[key_i].filter(to_keep, *key_iterable)
            return filtered
        else:
            filtered = {key: self[key] for key in to_keep}
            filtered = list(filtered.values()) if self._type is list else filtered
            return self.__class__(filtered)

    def get(self, *key_iterable):
        key_iterable = list(key_iterable)
        key_i = key_iterable.pop(0)
        if key_iterable:
            return self[key_i].get(*key_iterable)
        else:
            return self.contents[key_i]

    def __getitem__(self, key):
        if self.is_array(key) or is_slice(key):
            item = self._index_with_array_or_slice(key)
        elif self.is_container(key):
            item = self._index_with_container(key)
        # If given iterable, attempt to interpret it as an arraytainer:
        elif isinstance(key, (dict, list, tuple)):
            key = self.__class__(key)
            item = self._index_with_container(key)
        else:
            item = self._index_with_hash(key)
        return item

    def _index_with_container(self, key_container):        
        item = {key: self.contents[key][val] for key, val in key_container.items()}
        item = list(item.values()) if self._type is list else item
        return self.__class__(item)

    def _index_with_array_or_slice(self, array_or_slice):
        item = {key: self.contents[key][array_or_slice] for key in self.keys()}
        item = list(item.values()) if self._type is list else item
        return self.__class__(item)

    def _index_with_hash(self, key):
        try:
            item = self.contents[key]
        except KeyError:
            error_msg = (f'{key} is not a key in the {self.__class__.__name__};',
                         f'valid keys for this {self.__class__.__name__} are {tuple(self.keys())}.')
            raise KeyError(' '.join(error_msg))
        return item

class SetterMixin:

    def update(self, new_val, *key_iterable):
        if key_iterable:
            key_iterable = list(key_iterable)
            key_i = key_iterable.pop(0)
            self[key_i].update(new_val, *key_iterable)
        else:
            new_val = self.__class__(new_val)
            if self._type is dict:
                self.contents.update(new_val)
            else:
                self.contents.append(new_val)

    def __setitem__(self, key, new_value):

        # Ensure new_value is either an arraytainer or an array:
        if isinstance(new_value, Number):
            new_value = self.array(new_value)
        elif not self.is_array(new_value):
            new_value = self.__class__(new_value) 
        
        if self.is_array(key) or is_slice(key):
            self._set_with_array_or_slice(key, new_value)
        elif self.is_container(key):
            self._set_with_container(key, new_value)
        # Interpret indexing with list/dict as a container:
        elif isinstance(key, (dict, list, tuple)):
            key = self.__class__(key)
            item = self._set_with_container(key, new_value)
        else:
            self._set_with_hash(key, new_value) 

    def _set_with_container(self, key_container, new_value):
        for key, val in key_container.items():
            if self.is_array(self[key][val]):
                self._set_array_item(key, val, new_value)
            else:
                self[key][val] = new_value[key] if self.is_container(new_value) else new_value

    def _set_with_array_or_slice(self, array_or_slice, new_value):
        for container_key in self.keys():
            value_i = new_value[container_key] if self.is_container(new_value) else new_value
            self._set_array_item(container_key, array_or_slice, value_i)

    def _set_array_item(self, key, idx, value_i):
        error_msg = ('Cannot use array to set value with base Arraytainer class;', 
                     'use a Numpytainer or Jaxtainer instead.')
        return KeyError(' '.join(error_msg))

    def _set_with_hash(self, key, new_value):
        
        if not (self.is_container(new_value) or self.is_array(new_value)):
            new_value = self.__class__(new_value)

        try:
            self.contents[key] = new_value
        except IndexError:
            if self._type is list:
                error_msg = ("Unable to new assign items to a list-like container;",
                             "use the append method instead.")
                raise TypeError(' '.join(error_msg))
            else:
                raise IndexError(f"Provided key {key} is invalid.")