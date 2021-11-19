from itertools import chain

class Mixin:
    
    def keys(self):
        if self._type is dict:
            keys = [key for key in self.contents.keys()]
            keys = sort_keys(keys)
        else:
            keys = tuple(i for i in range(len(self.contents)))
        return keys

    def values(self):
        return tuple(self[key] for key in self.keys())

    def items(self):
        return tuple((key, self[key]) for key in self.keys())

# Sorts keys of uncomparable types in arbitrary (but consistent) manner:
def sort_keys(keys):
    types_in_list = [(type(key).__name__) for key in keys]
    types_in_list = list(set(types_in_list))
    types_in_list = sorted(types_in_list)
    sorted_sublists = [sorted([key for key in keys if type(key).__name__ == type_i]) for type_i in types_in_list]
    sorted_keys = tuple(chain.from_iterable(sorted_sublists))
    return sorted_keys