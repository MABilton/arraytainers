from more_itertools.more import always_iterable
from .base import Arraytainer

class Mixin:
    
    def __getitem__(self, key):
        key_type = type(key) 
        # If indexing with an array container:
        if issubclass(key_type, Arraytainer):
            item = self._index_with_container(key)
        # If indexing with an array:
        elif key_type in self.ARRAY_TYPES:
            item = self._index_with_array(key)
        # If indexing using a slice or tuple of slices:
        elif all(isinstance(val, slice) for val in always_iterable(key)):
            item = self._index_with_slices(key)
        # If we're given a tuple of integers:
        elif isinstance(key, tuple) and all(isinstance(val, int) for val in always_iterable(key)):
            self._set_with_slices(key, new_value)
        # Index using a regular hash:
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

    def array(self, in_array):
        error_msg = '''Base Arraytainer class does not have an array method; 
                    instead, use a Numpytainer of Jaxtainer.'''
        return TypeError(error_msg)

    def _index_with_hash(self, key):
        try:
            item = self.contents[key]
        # Hash passed could be an integer instead of a key (e.g. by a Numpy function)
        except KeyError:
            keys = self.keys()
            item = self.contents[keys[key]]
        return item