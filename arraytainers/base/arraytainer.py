
import numpy as np

class Arraytainer:

    _arrays = (ndarray,)

    def __init__(self, contents):
        self._contents = self._preprocess_contents(contents)
        self._type = dict if hasattr(self.contents, 'keys') else list

    @classmethod
    def from_vector(cls, vector, shapes, order='C'):

        try:
            vector = vector.flatten(order=order).tolist()
        except AttributeError:
            raise ValueError('Only arrays which can be converted to lists are '
                             'allowed to be passed to the from_vector constructor.')

        shapes = cls(shapes, greedy_array_conversion=True) 
        new_contents = cls._create_arraytainer_from_vector(vector, shapes, order)

        return cls(new_contents)
    
    @staticmethod
    def _create_arraytainer_from_vector(vector, shapes, order, new_arraytainer=None):
        
        if new_arraytainer is None:
            new_arraytainer = shapes.copy().unpacked

        for key, shape in shapes.items():
            if isinstance(shapes, Arraytainer):
                num_vals = np.prod(shape)
                vals_i = np.array([vector.pop(0) for _ in range(num_vals)])
                new_arraytainer[key] = vals_i.reshape(shape, order=order)
            else:
                new_arraytainer[key] = \
                    Arraytainer._create_arraytainer_from_vector(vector, shape, order, output=new_arraytainer[key])
        
        return new_arraytainer

    @staticmethod
    def _preprocess_contents(contents):

        if hasattr(contents, 'keys'):
            self._check_keys(contents)

        if isinstance(contents, self._arrays):
            contents = [contents]
        elif isinstance(contents, Arraytainer):
            contents = contents.unpack()
        elif not hasattr(contents, '__len__'):
            contents = [contents]
        elif isinstance(contents, tuple):
            contents = list(contents)

        return deepcopy(contents)

    @staticmethod
    def _check_keys(contents):
        for key in contents.keys():
            if isinstance(key, tuple):
                raise KeyError(f'Contents contains the tuple key {key}, which are not allowed in Arraytainers.')

    def __len__(self):
        return len(self.contents)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.unpacked)})"

    def copy(self):
        return self.__class__(deepcopy(self.unpacked))

    def keys(self):
        if self._type is dict:
            keys = (key for key in self.contents.keys())
        else:
            keys = (i for i in range(len(self.contents)))
        return keys

    def values(self, unpacked=False):
        contents = self.unpacked if unpacked else self
        return (contents[key] for key in self.keys())

    def items(self, unpacked=False):
        contents = self.unpacked if unpacked else self
        return ((key, contents[key]) for key in self.keys())

    def __iter__(self):
        return self.values()

    def unpack(self):
        output = [val.unpack() if isinstance(val, Arraytainer) else val for val in self.values()]
        if self._type is dict:
            output = dict(zip(self.keys(), output))
        return output

    @property
    def unpacked(self):
        return self.unpack()

    def list_arrays(self):
        unpacked = self.unpacked
        return self._flatten_contents(unpacked)
    
    @property
    def array_list(self):
        return self.list_arrays()
    
    @staticmethod
    def _flatten_contents(contents, array_list=None):
        
        if array_list is None:
            array_list = []
        
        if isinstance(contents, list):
            keys = range(len(contents))  
        else:
            keys = contents.keys()

        for key in keys:
            if isinstance(contents[key], (dict,list)):
                array_list = Arraytainer._flatten_contents(contents[key], array_list)
            else:
                array_list.append(contents[key])
        
        return array_list


