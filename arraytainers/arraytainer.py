from ._contents import Contents
import copy
import numbers
import numpy as np
import more_itertools

class Arraytainer(Contents, np.lib.mixins.NDArrayOperatorsMixin):

    _arrays = (np.ndarray,)

    #
    #   Constructor Methods
    #

    def __init__(self, contents, convert_arrays=True, greedy=False, nested=True):
        
        # All arraytainers must minimally comprise of a list:
        if not isinstance(contents, (list, dict, tuple, Arraytainer)):
            contents = [contents]
            # To prevent array converter converting outer list to array:
            greedy=True

        if convert_arrays:
            contents = self._convert_contents_to_arrays(contents, greedy)

        super().__init__(contents, nested, greedy=greedy) #convert_arrays=convert_arrays

    @classmethod
    def from_array(cls, array, shapes, order='C', convert_arrays=True, greedy=False, nested=True):
        
        # Concatenate shapes tuple into a single arraytainer:
        if not isinstance(shapes, tuple):
            shapes = tuple([shapes])
        shapes = cls._concatenate_elements_to_array(shapes)

        if not isinstance(shapes, Arraytainer):
            raise ValueError('shapes must container at least one arraytainer.')
            
        # Ensure correct number of elements in array:
        total_size = np.prod(shapes).sum_all()
        if total_size != array.size:
            raise ValueError(f'Array contains {array.size} elements, but shapes '
                             f'contains {total_size} elements.')

        vector = array.flatten(order=order)
        contents = cls._create_contents_from_vector(vector, shapes, order)

        return cls(contents, convert_arrays, greedy, nested)

    @classmethod
    def _concatenate_elements_to_array(cls, val_tuple):
        
        val_list = list(val_tuple)

        for idx, val_i in enumerate(val_list):
            
            if not isinstance(val_i, (Arraytainer, *cls._arrays)):
                val_i = cls._convert_to_array(val_i)
                # Shape arrays must have at least one dimension for concatenate:
                if val_i.ndim == 0:
                    val_i = val_i[None]

            elif isinstance(val_i, Arraytainer):
                # 0 dimensional arrays in arraytainer cause concatenate to throw errors:
                val_i = np.atleast_1d(val_i)

            val_list[idx] = val_i

        return np.concatenate(val_list)

    @classmethod
    def _create_contents_from_vector(cls, vector, shapes, order, elem_idx=None, first_call=True):

        if elem_idx is None:
            elem_idx = 0

        new_contents = {}
        for key, shape in shapes.items():
            if isinstance(shape, Arraytainer):
                new_contents[key], elem_idx = cls._create_contents_from_vector(vector, shape, order, elem_idx, first_call=False)
            else:
                array_vals, elem_idx = cls._extract_array_vals(vector, elem_idx, shape)
                new_contents[key] = array_vals.reshape(shape, order=order)

        if shapes._type is list:
            new_contents = list(new_contents.values())

        if first_call:
            output = new_contents
        else:
            output = (new_contents, elem_idx)
        
        return output
 
    @staticmethod
    def _extract_array_vals(vector, elem_idx, shape):
        # Jaxtainer version of this method uses jnp methods instead of np:
        num_elem = np.prod(shape)
        array_vals = vector[elem_idx:elem_idx+num_elem]
        elem_idx += num_elem
        return array_vals, elem_idx

    #
    #   Array Conversion Methods
    #

    @staticmethod
    def _convert_to_array(val):
        # Note that Jaxtainer uses jnp.array:
        return np.array(val)

    def _convert_contents_to_arrays(self, contents, greedy):

        contents = self._unpack_if_arraytainer(contents)
        if isinstance(contents, tuple):
            contents = list(contents)

        contents_iter = contents.items() if isinstance(contents, dict) else enumerate(contents)

        for key, val in contents_iter:
            val = self._unpack_if_arraytainer(val)

            if isinstance(val, dict):
                contents[key] = self._convert_contents_to_arrays(val, greedy)

            elif isinstance(val, (list, tuple)):
                
                # Check before altering contents on val:
                any_arrays_in_val = any(isinstance(val_i, self._arrays) for val_i in val)

                # List of numbers should be directly converted to an array:
                if all(isinstance(val_i, numbers.Number) for val_i in val):
                    converted_vals = self._convert_to_array(val)
                else:
                    converted_vals = self._convert_contents_to_arrays(val, greedy)
            
                if not greedy and self._can_combine_list_into_single_array(converted_vals, any_arrays_in_val):
                    converted_vals = self._convert_to_array(converted_vals)

                contents[key] = converted_vals

            else:
                contents[key] = self._convert_to_array(val)

        return contents

    @staticmethod
    def _unpack_if_arraytainer(val):
        if isinstance(val, Arraytainer):
            val = val.unpack()
        return val

    def _can_combine_list_into_single_array(self, converted_list, any_arrays_in_val):
        can_convert = not any_arrays_in_val and \
                      self._all_elems_are_arrays(converted_list) and \
                      self._all_arrays_of_equal_shape(converted_list)
        return can_convert

    def _all_elems_are_arrays(self, converted_list):
        return all(isinstance(val_i, self._arrays) for val_i in converted_list)

    def _all_arrays_of_equal_shape(self, converted_list):
        # Converted list could be empty:
        if converted_list:
            first_array_shape = converted_list[0].shape
            all_equal = all(array_i.shape == first_array_shape for array_i in converted_list)
        else:
            all_equal = True
        return all_equal

    #
    #   Getter Methods
    #

    def __getitem__(self, key):
        if isinstance(key, self._arrays) or self._is_slice(key):
            item = self._get_with_array(key)
        else:
            item = super().__getitem__(key, greedy=True)
        return item
    
    def _get_with_array(self, array_key):
        item = {key: self._contents[key][array_key] for key in self.keys()}
        if self._type is list:
            item = list(item.values())
        return self.__class__(item, greedy=True)

    #
    #   Setter Methods
    #

    def __setitem__(self, key, new_value):
        if isinstance(key, self._arrays) or self._is_slice(key):
            self._set_with_array(key)
        elif isinstance(key, Arraytainer):
            self._set_with_arraytainer(key, new_value)
        else:
            super().__setitem__(key, new_value)

    @staticmethod
    def _is_slice(val):
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

    def _set_with_array(self, array_key, new_value):
        for key in self.keys():
            value_i = new_value[key] if isinstance(new_value, Arraytainer) else new_value
            # Note that Jaxtainers use different _set_array_values method:
            self._set_array_values(key, array_key, value_i)

    def _set_array_values(self, key, idx, new_value):
        self._contents[key][idx] = new_value

    def _set_with_arraytainer(self, arraytainer_key, new_value):
        for key, val in arraytainer_key.items():
            new_value_i = new_value[key] if isinstance(new_value, Arraytainer) else new_value
            if isinstance(val, self._arrays):
                self._set_array_values(key, val, new_value_i)
            else:
                self._contents[key][val] = new_value_i

    #
    #   Array Methods and Properties
    #

    @property
    def T(self):
        return np.transpose(self)
    
    def all(self):
        for key in self.keys():
            # Numpy/Jax arrays also have an 'all' method:
            if not self.contents[key].all():
                return False
        return True

    def any(self):
        for key in self.keys():
            if self.contents[key].any():
                return True
        return False

    def sum(self):
        return sum(self.values())

    def sum_arrays(self):
        return sum(self.list_elements())

    def sum_all(self):
        arraytainer_of_scalars = np.sum(self)
        return sum(arraytainer_of_scalars.list_elements())

    def get_shape(self, return_tuples=False):
        
        shapes = {}
        for key, val in self.items():
            shapes[key] = val.shape
        
        if self._type is list:
            shapes = list(shapes.values())

        if not return_tuples:
            shapes = self.__class__(shapes, greedy=True)

        return shapes

    #
    #   Array-Like Methods
    #

    @property
    def shape(self):
        return self.get_shape()

    @property
    def ndim(self):
        return np.ndim(self)

    @property
    def sizes(self):
        return np.prod(self.shape)

    @property
    def size(self):
        size = self.sizes.sum_all()
        if isinstance(size, self._arrays):
            size = size.item()
        return size

    def reshape(self, *new_shapes, order='C'):
        new_shapes = self._concatenate_elements_to_array(new_shapes)
        return np.reshape(self, new_shapes, order=order)

    def flatten(self, order='C', return_array=True):
        output = np.squeeze(np.ravel(self, order=order))
        if return_array:
            # Zero-dimensional elements cause concatenate to throw error:
            elem_list = output.list_elements()
            for idx, elem in enumerate(elem_list):
                if elem.ndim == 0:
                    elem_list[idx] = elem[None]
            output = np.concatenate(elem_list)
        return output

    #
    #   Numpy Function Handling Methods
    #

    def __array_function__(self, func, types, args, kwargs):
        fun_return = self._manage_func_call(func, types, *args, **kwargs)
        return fun_return

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        fun_return = self._manage_func_call(ufunc, method, *args, **kwargs)
        return fun_return

    def _manage_func_call(self, func, types, *args, **kwargs):

        arraytainer_list = self._list_arraytainers_in_args(args) + self._list_arraytainers_in_args(kwargs)
        largest_arraytainer = self._find_largest_arraytainer(arraytainer_list)
        self._check_arraytainer_arg_compatability(arraytainer_list, largest_arraytainer)
        shared_keyset = self._get_shared_keyset(arraytainer_list)

        func_return = copy.deepcopy(largest_arraytainer.contents)
        for key in shared_keyset:
            args_i = self._prepare_func_args(args, key)
            kwargs_i = self._prepare_func_args(kwargs, key)
            func_return[key] = func(*args_i, **kwargs_i)

        return self.__class__(func_return, greedy=True)

    @staticmethod
    def _list_arraytainers_in_args(args):
        args_iter = args.values() if isinstance(args, dict) else args
        args_arraytainers = []
        for arg_i in args_iter:
            if isinstance(arg_i, Arraytainer):
                args_arraytainers.append(arg_i)
            # Could have an arraytainer in a list in a tuple (e.g. np.concatenate):
            elif isinstance(arg_i, (list, tuple, dict)):
                args_arraytainers += Arraytainer._list_arraytainers_in_args(arg_i)
        
        return args_arraytainers

    @staticmethod
    def _find_largest_arraytainer(arraytainer_list):
        largest_len = -1
        for arraytainer in arraytainer_list:
            if len(arraytainer) > largest_len:
                largest_len = len(arraytainer)
                largest_arraytainer = arraytainer
        return largest_arraytainer

    def _check_arraytainer_arg_compatability(self, arraytainer_list, largest_arraytainer):

        largest_keyset = set(largest_arraytainer.keys())
            
        for arraytainer_i in arraytainer_list:
            
            if arraytainer_i._type != largest_arraytainer._type:
                raise KeyError('Arraytainers being combined through operations must',
                               'be all dictionary-like or all list-like')
            
            ith_keyset = set(arraytainer_i.keys())
            if not ith_keyset.issubset(largest_keyset):
                raise KeyError(f'Keys of an Arraytainer (= {ith_keyset}) is not a subset of the ',
                               f"keys of a larger Arraytainer (={largest_keyset}) it's being combined with.")
    
    @staticmethod
    def _get_shared_keyset(arraytainer_list):
        return set.intersection(*[set(arraytainer.keys()) for arraytainer in arraytainer_list])


    @staticmethod
    def _prepare_func_args(args, key):
        
        args_iter = args.items() if isinstance(args, dict) else enumerate(args)
        
        prepped_args = {}
        for arg_key, arg in args_iter:
            if isinstance(arg, Arraytainer):
                # Skip over this arg if doesn't include key:
                if key in arg.keys():
                    prepped_args[arg_key]  = arg[key]
            # Tuple args may contain arryatainer entries:
            elif isinstance(arg, (tuple, list, dict)):
                prepped_args[arg_key] = Arraytainer._prepare_func_args(arg, key)
            else:
                prepped_args[arg_key] = arg

        if not isinstance(args, dict):
            prepped_args = tuple(prepped_args.values())

        return prepped_args