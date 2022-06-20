import warnings
import copy
from more_itertools import always_iterable
import numbers
import numpy as np

class Arraytainer(np.lib.mixins.NDArrayOperatorsMixin):

    def __init__(self, contents, array_conversions=True, dtype=None):
        if not isinstance(contents, (list, dict)):
            raise ValueError(f'contents must be either a list or a dict, not a {type(contents)}.')
        contents = self._deepcopy_contents(contents)
        if isinstance(contents, dict) and self._keys_contain_tuple(contents):
            raise KeyError("contents.keys() contains a tuple, which isn't allowed in an Arraytainer.")
        contents = self._recursive_arraytainer_conversion(contents, dtype, array_conversions)
        self._contents = contents
        self._dtype = dtype

    @staticmethod
    def _keys_contain_tuple(contents):
        keys = contents.keys()
        contains_tuple = False
        for key in always_iterable(keys):
            if isinstance(key, tuple):
                contains_tuple = True
                break
        return contains_tuple

    #
    #   Alternative Constructors
    #

    @classmethod
    def from_array(cls, array, shapes, order='C'):
        # Concatenate shapes tuple into a single arraytainer
        shapes = cls.by_concatenation(shapes)
        # Ensure correct number of elements in array:
        total_size = shapes.size
        if total_size != array.size:
            raise ValueError(f"""Array contains {array.size} elements, but shapes 
                                 contains {total_size} elements.""")
        vector = array.flatten(order=order)
        contents, _ = cls._create_contents_from_vector(vector, shapes, order)
        return cls(contents)

    @classmethod
    def _create_contents_from_vector(cls, vector, shapes, order, vector_idx=None):
        if vector_idx is None:
            vector_idx = 0
        contents = {}
        for key, shape in shapes.items():
            if isinstance(shape, Arraytainer):
                contents[key], vector_idx = cls._create_contents_from_vector(vector, shape, order, vector_idx)
            else:
                num_elem = shape.prod(dtype=int)
                array_vals = vector[vector_idx:vector_idx+num_elem]
                vector_idx = vector_idx + num_elem
                new_contents[key] = array_vals.reshape(shape, order=order)
        if shapes.contents_type is list:
            contents = list(contents.values())
        return contents, vector_idx

    @classmethod
    def by_concatenation(cls, values, dtype=None):
        if not isinstance(values, (list, tuple)):
            values = [values]
        val_is_arraytainer = [isinstance(val, Arraytainer) for val in values]
        if not any(val_is_arraytainer):
            raise ValueError('Must contain at least one arraytainer.')
        to_concat = []
        for idx, val_i in enumerate(values):
            if isinstance(val_i, Arraytainer):
                val_i = np.atleast_1d(val_i)
            elif not cls.is_array(val_i):
                val_i = cls.create_array(val_i, dtype)
                # Shape arrays must have at least one dimension for concatenate:
                if val_i.ndim == 0:
                    val_i = val_i[None]
            to_concat.append(val_i)
        concatenated = np.concatenate(to_concat)
        return concatenated

    def __len__(self):
        return len(self._contents)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.unpack())})"

    def __iter__(self):
        return self.values()

    def __getitem__(self, key):
        if isinstance(key, Arraytainer):
            item = self._get_with_arraytainer(key)
        elif self.is_array(key) or self.is_slice(key):
            item = self._get_with_array(key)
        else:
            item = self._get_with_regular_key(key)
        return item

    def __setitem__(self, key, new_val):
        new_val = self._convert_to_array_or_arraytainer(new_val)
        if self.is_array(key) or self.is_slice(key):
            self._set_with_array(key, new_val)
        elif isinstance(key, Arraytainer):
            self._set_with_arraytainer(key, new_val)
        else:
            self._set_with_regular_key(key, new_val)

    def __array_function__(self, func, types, args, kwargs):
        fun_return = self._manage_func_call(func, types, *args, **kwargs)
        return fun_return

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        fun_return = self._manage_func_call(ufunc, method, *args, **kwargs)
        return fun_return

    def copy(self):
        return copy.copy(self)

    def deepcopy(self):
        return copy.deepcopy(self)

    @staticmethod
    def _deepcopy_contents(contents):
        return copy.deepcopy(contents)

    def keys(self):
        if self.contents_type is dict:
            keys = self._contents.keys()
        else:
            keys = range(len(self._contents))
        return keys

    def values(self):
        return (self._contents[key] for key in self.keys())

    def items(self):
        return ((key, self._contents[key]) for key in self.keys())

    def append(self, new_val, *key_iterable):
        if self.contents_type is not list:
            raise AttributeError("""Can't append to dictionary-like Arraytainer; 
            use update method instead.""") 
        new_val = self._convert_to_array_or_arraytainer(new_val)
        key_iterable = list(key_iterable)
        key = key_iterable.pop(0)
        if key_iterable:
            self[key].append(new_val, *key_iterable)
        else:
            self._contents.append(new_val)
                
    def update(self, new_val, *key_iterable):
        if self.contents_type is not dict:
            raise TypeError('''Can't update a list-like Arraytainer; 
            use the append method instead.''')
        new_val = self._convert_to_array_or_arraytainer(new_val)
        key_iterable = list(key_iterable)
        key = key_iterable.pop(0)
        if key_iterable:
            self[key].update(new_val, *key_iterable)
        else:
            self[key]._contents.update(new_val)

    def _get_with_arraytainer(self, arraytainer_key):
        item = {}
        for key, val in arraytainer_key.items():
            if key not in self.keys():
                raise KeyError("Arraytainer index contains keys not found in original Arraytainer.")
            if self.is_array(self._contents[key]) and isinstance(val, Arraytainer):
                raise KeyError("Arraytainer index contains keys not found in original Arraytainer.")
            item[key] = self._contents[key][val]
        if self.contents_type is list:
            item = list(item.values())
        return self.__class__(item)   

    def _get_with_array(self, array_key):
        item = {key: self._contents[key][array_key] for key in self.keys()}
        if self.contents_type is list:
            item = list(item.values())
        return self.__class__(item)

    def _get_with_regular_key(self, key):
        try:
            item = self._contents[key]
        except (IndexError, KeyError):
            key_strs = [str(key_i) for key_i in self.keys()]
            raise KeyError(f"""{key} is not a key in this Arraytainer; valid keys are: {', '.join(key_strs)}.""")        
        return item

    @property
    def contents(self):
        return self._contents

    @property
    def contents_type(self):
        return type(self._contents)

    @property
    def sizes(self):
        return np.prod(self.shape, dtype=int)

    @property
    def first_array(self):
        return self.get_first_array()

    def unpack(self):
        unpacked = []
        for val in self.values():
            if isinstance(val, Arraytainer):
                unpacked.append(val.unpack())
            else:
                unpacked.append(val)
        if self.contents_type is dict:
            unpacked = dict(zip(self.keys(), unpacked))
        return unpacked

    def list_arrays(self):
        return self._recursively_get_arrays()

    def get_first_array(self):
        first_val = next(self.values())
        if isinstance(first_val, Arraytainer):
            first_val = first_val.get_first_array()
        return first_val

    def _recursively_get_arrays(self, array_list=None):
        if array_list is None:
            array_list = []
        for val in self.values():
            if isinstance(val, Arraytainer):
                array_list = val._recursively_get_arrays(array_list)
            else:
                array_list.append(val)
        return array_list
    
    def all(self):
        for key in self.keys():
            if not self._contents[key].all():
                return False
        return True

    def any(self):
        for key in self.keys():
            if self._contents[key].any():
                return True
        return False

    def sum_elems(self):
        return sum(self.values())

    def sum_arrays(self):
        return sum(self.list_arrays())

    def sum(self):
        arraytainer_of_scalars = np.sum(self)
        return sum(arraytainer_of_scalars.list_arrays())

    def reshape(self, new_shapes, order='C'):
        if not isinstance(new_shapes, (tuple, list)):
            new_shapes = (new_shapes,)
        is_arraytainer = [isinstance(shape, Arraytainer) for shape in new_shapes]
        if any(is_arraytainer) and not all(is_arraytainer):
            new_shapes = self.__class__.by_concatenation(new_shapes)
        return np.reshape(self, new_shapes, order=order)

    def flatten(self, order='C', return_array=True):
        output = np.squeeze(np.ravel(self, order=order))
        if return_array:
            # Zero-dimensional elements cause concatenate to throw error:
            elem_list = output.list_arrays()
            for idx, elem in enumerate(elem_list):
                if elem.ndim == 0:
                    elem_list[idx] = elem[None]
            output = np.concatenate(elem_list)
        return output

    def tolist(self):
        output = {}
        for key, val in self.items():
            if self.is_array(val) or isinstance(val, Arraytainer):
                output[key] = val.tolist()
            else:
                output[key] = val
        if self.contents_type is list:
            output = list(output.values())
        return output

    @property
    def T(self):
        return np.transpose(self)

    @property
    def shape(self):
        shapes = {}
        for key, val in self.items():
            shapes[key] = val.shape
        if self.contents_type is list:
            shapes = list(shapes.values())
        return self.__class__(shapes)

    @property
    def ndim(self):
        return np.ndim(self)

    @property
    def size(self):
        return self.sizes.sum_all()

    @property
    def dtype(self):
        return self._dtype

    def _set_with_array(self, array, new_val):
        if isinstance(new_val, Arraytainer):
            for key in new_val.keys():
                if (key not in self.keys()):
                    raise KeyError('New Arraytainer value to set contains keys not found in the original Arraytainer.')
                if self.is_array(self._contents[key]) and (not self.is_array(new_val[key])):
                    raise KeyError('New Arraytainer value to set contains keys not found in the original Arraytainer.')
                elif self.is_array(self._contents[key]):
                    self._set_array_values(key, array, new_val[key])      
                else:
                    self[key][array] = new_val[key]
        else:
            for key in self.keys():
                if self.is_array(self._contents[key]):
                    self._set_array_values(key, array, new_val) 
                else:
                    self[key][array] = new_val

    def _set_array_values(self, key, mask, new_value):
        self._contents[key][mask] = new_value

    def _set_with_arraytainer(self, arraytainer, new_val):
        for key, mask in arraytainer.items():
            self_val = self._contents[key]
            if isinstance(self_val, Arraytainer) and isinstance(new_val, Arraytainer):
                self._contents[key][mask] = new_val._contents[key]
            elif isinstance(self_val, Arraytainer) and self.is_array(new_val):
                self._contents[key][mask] = new_val
            elif self.is_array(self_val) and self.is_array(new_val):
                self._set_array_values(key, mask, new_val)
            elif self.is_array(self_val) and isinstance(new_val, Arraytainer):                
                if (key not in new_val.keys()) or (not self.is_array(new_val[key])):
                    raise KeyError('New Arraytainer value to set contains keys not found in the original Arraytainer.')
                self._set_array_values(key, mask, new_val[key]) 
                

    def _set_with_regular_key(self, key, new_val):
        try:
            self._contents[key] = new_val
        except IndexError as e:
            if self.contents_type is list:
                raise KeyError("""Unable to new assign items to a list-like Arraytainer;
                                use the append method instead.""")
            raise e

    def _convert_to_array_or_arraytainer(self, val):
        if not (self.is_array(val) or isinstance(val, Arraytainer)):
            if isinstance(val, (list, dict)):
                val = self.__class__(val, dtype=self._dtype)
            else:
                val = self.create_array(val, dtype=self._dtype)
        return val


    def _manage_func_call(self, func, types, *args, **kwargs):
        arraytainer_list = self._list_arraytainers_in_args(args) + self._list_arraytainers_in_args(kwargs)
        largest_arraytainer = max(arraytainer_list, key=len)
        self._check_arg_compatability(arraytainer_list, largest_arraytainer)
        shared_keyset = self._get_shared_keyset(arraytainer_list)
        func_return = copy.deepcopy(largest_arraytainer.contents)
        for key in shared_keyset:
            args_i = self._prepare_func_args(args, key)
            kwargs_i = self._prepare_func_args(kwargs, key)
            func_return[key] = func(*args_i, **kwargs_i)
        return self.__class__(func_return)

    def _list_arraytainers_in_args(self, args):
        args_iter = args.values() if isinstance(args, dict) else args
        args_arraytainers = []
        for arg_i in args_iter:
            if isinstance(arg_i, Arraytainer):
                args_arraytainers.append(arg_i)
            # Could have an arraytainer in a list in a tuple (e.g. np.concatenate):
            elif isinstance(arg_i, (list, tuple, dict)):
                args_arraytainers += self._list_arraytainers_in_args(arg_i)
        return args_arraytainers

    def _check_arg_compatability(self, arraytainer_list, largest_arraytainer):
        largest_keyset = set(largest_arraytainer.keys())
        for arraytainer in arraytainer_list:
            if arraytainer.contents_type != largest_arraytainer.contents_type:
                raise KeyError("""Arraytainers being combined through operations 
                must be all dictionary-like or all list-like""")
            keyset = set(arraytainer.keys())
            if not keyset.issubset(largest_keyset):
                raise KeyError(f"""Keys of arraytainer argument (= {keyset}) are not 
                a subset of the keys of the largest Arraytainer (={largest_keyset}) it's being combined with.""")
    
    @staticmethod
    def _get_shared_keyset(arraytainer_list):
        return set.intersection(*[set(arraytainer.keys()) for arraytainer in arraytainer_list])

    def _prepare_func_args(self, args, key):
        args_iter = args.items() if isinstance(args, dict) else enumerate(args)
        prepped_args = {}
        for arg_key, arg in args_iter:
            if isinstance(arg, Arraytainer):
                # Skip over this arg if doesn't include key:
                if key in arg.keys():
                    prepped_args[arg_key] = arg[key]
            # Tuple args may contain arryatainer entries:
            elif isinstance(arg, (tuple, list, dict)):
                prepped_args[arg_key] = self._prepare_func_args(arg, key)
            else:
                prepped_args[arg_key] = arg
        if not isinstance(args, dict):
            prepped_args = tuple(prepped_args.values())
        return prepped_args

    @staticmethod
    def is_slice(val):
        if isinstance(val, slice):
            is_slice = True
        # Slices accross multiple dimensions appear as tuples of ints/slices/Nones (e.g. my_array[3, 1:2, :])
        elif isinstance(val, tuple) and all(isinstance(val_i, (type(None), slice, int)) for val_i in val):
            is_slice = True
        elif val is None:
            is_slice = True
        else:
            is_slice = False
        return is_slice

    @staticmethod
    def is_array(val):
        return isinstance(val, np.ndarray)

    @staticmethod
    def create_array(val, dtype=None):
        return np.array(val, dtype=dtype)

    def _recursive_arraytainer_conversion(self, contents, dtype, array_conversions=True):
        items = contents.items() if isinstance(contents, dict) else enumerate(contents)
        for key, val in items:
            if isinstance(val, Arraytainer):
                continue
            elif isinstance(val, (list, dict)):
                contents[key] = self.__class__(val, array_conversions, dtype)
            elif array_conversions:
                contents[key] = self.create_array(val, dtype)
        return contents

    def squeeze(self, axis=None):
        return np.squeeze(self, axis=axis)

    def swapaxes(self, axis1, axis2):
        return np.swapaxes(self, axis1, axis2)

    def transpose(self):
        return np.transpose(self)

    def resize(self, new_shapes):
        if not isinstance(new_shapes, (tuple, list)):
            new_shapes = (new_shapes,)
        is_arraytainer = [isinstance(shape, Arraytainer) for shape in new_shapes]
        if any(is_arraytainer) and not all(is_arraytainer):
            new_shapes = self.__class__.by_concatenation(new_shapes)
        return np.resize(self, new_shapes)