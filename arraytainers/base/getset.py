
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

    def get(self, *key_iterable):
        key_iterable = list(key_iterable)
        key_i = key_iterable.pop(0)
        if key_iterable:
            return self[key_i].get(*key_iterable)
        else:
            return self.contents[key_i]

    def __getitem__(self, key):
        if self.is_container(key):
            item = self._index_with_container(key)
        # Interpret indexing with list/dict as a container:
        elif isinstance(key, (dict, list)):
            key = self.__class__(key)
            item = self._index_with_container(key)
        elif self.is_array(key):
            item = self._index_with_array(key)
        elif is_slice(key):
            item = self._index_with_slices(key)
        # Treat tuple which isn't a slice as a tuple of non-array keys:
        elif isinstance(key, tuple):
            item = {key_i: self[key_i] for key_i in key}
            item = list(item.values()) if self._type is list else item
            item = self.__class__(item)
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

def _attempt_append(contents, new_val):
    try:
        contents.append(new_val)
    except AttributeError:
        error_msg = ('Unable to append values to dictionary-like container;',
                    "use 'my_arraytainer[new_key] = new_value' or the my_arraytainer.set method instead.")
        raise AttributeError(' '.join(error_msg))

class SetterMixin:
    
    def append(self, new_val, key_iterable=()):
        if key_iterable:
            key_iterable = list(key_iterable)
            key_i = key_iterable.pop(0)
            if key_iterable:
                self[key_i].append(new_val, key_iterable)
            else:
                _attempt_append(self.contents[key_i], new_val)
        else:
            _attempt_append(self.contents, new_val) 

    # def set(self, new_val, *key_iterable):

    #     if not key_iterable:
    #         raise KeyError('Must specify at least one key when using the set method.')

    #     # If key_iterable is provided by the user as a list/tuple:
    #     if len(key_iterable) == 1 and isinstance(key_iterable[0], (tuple,list)):
    #         key_iterable = key_iterable[0]
            
    #     key_iterable = list(key_iterable)
    #     key_i = key_iterable.pop(0)

    #     if key_iterable:
    #         self[key_i].set(new_val, *key_iterable)
    #     else:
    #         try:
    #             self[key_i] = new_val
    #         except AttributeError:
    #             error_msg = ("Unable to new assign items to a list-like container;",
    #                          "use the append method instead.")
    #             raise AttributeError(' '.join(error_msg))

    def __setitem__(self, key, new_value):

        new_value = self._preprocess_set_new_value(new_value)
        
        if self.is_container(key):
            self._set_with_container(key, new_value)
        # Interpret indexing with list/dict as a container:
        elif isinstance(key, (dict,list)):
            key = self.__class__(key)
            item = self._set_with_container(key, new_value)
        elif self.is_array(key):
            self._set_with_array(key, new_value)
        elif is_slice(key):
            self._set_with_slices(key, new_value)
        else:
            self._set_with_hash(key, new_value)    

    def _preprocess_set_new_value(self, new_val):
        if isinstance(new_val, Number):
            new_val = self.array(new_val)
        elif not self.is_array(new_val):
            new_val = self.__class__(new_val) 
        return new_val

    def _set_with_container(self, container_key, new_value):
        for key in self.keys():
            value_i = new_value[key] if self.is_container(new_value) else new_value
            idx_i = container_key[key]
            if self.is_array(self[key]):
                self._set_array_item(key, idx_i, value_i)
            else:
                self[key][idx_i] = value_i

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