from ._contents import Contents
import numpy as np
import more_itertools

class Arraytainer(Contents, np.lib.mixins.NDArrayOperatorsMixin):

    _arrays = (np.ndarray,)

    #
    #   Constructor Methods
    #

    def __init__(self, contents, convert_arrays=True, greedy=False, nested=True):
        if convert_arrays:
            contents = self._convert_contents_to_arrays(contents, greedy)
        super().__init__(contents, nested)
    

    @classmethod
    def from_array(cls, array, shapes, order='C'):

        vector = array.flatten(order=order).tolist()
        if not isinstance(shapes, Arraytainer):
            shapes = cls(shapes, greedy=True)

        contents = cls._create_contents_from_vector(vector, shapes, order)

        return cls(contents)

    @classmethod
    def _create_contents_from_vector(cls, vector, shapes, order, output=None):

        if output is None:
            output = shapes.copy().unpack()

        for key, shape in shapes.items():
            if isinstance(shape, Arraytainer):
                output[key] = cls._create_contents_from_vector(vector, shape, order, output=output[key])
            else:
                array_vals = cls._extract_array_vals(vector, shape)
                output[key] = array_vals.reshape(shape, order=order)

        return output
 
    @staticmethod
    def _extract_array_vals(vector, shape):
        # Jaxtainer version of this method uses jnp methods instead of np:
        num_elem = np.prod(shape)
        return np.array([vector.pop(0) for _ in range(num_elem)])

    #
    #   Array Conversion Methods
    #

    @staticmethod
    def _convert_to_array(val):
        # Note that Jaxtainer uses jnp.array:
        return np.array(val)

    def _convert_contents_to_arrays(self, contents, greedy, initial=True):

        contents = self._unpack_if_arraytainer(contents)
        if isinstance(contents, tuple):
            contents = list(contents)

        contents_iter = contents.items() if isinstance(contents, dict) else enumerate(contents)

        for key, val in contents_iter:

            val = self._unpack_if_arraytainer(val)
            
            if isinstance(val, dict):
                contents[key] = self._convert_contents_to_arrays(val, greedy)

            elif isinstance(val, (list, tuple)):

                converted_vals = self._convert_contents_to_arrays(val, greedy)

                if all(self._is_scalar_array(val) for val in converted_vals):
                    converted_vals = self._convert_to_array(converted_vals)

                if greedy and self._can_combine_list_into_single_array(converted_vals):
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

    @staticmethod
    def _is_scalar_array(val):
        return not val.shape

    def _can_combine_list_into_single_array(self, contents_list):

        elem_is_array = [isinstance(val_i, self._arrays) for val_i in contents_list]

        if all(elem_is_array):
            first_array_len = len(contents_list[0])
            is_equal_length = [len(array_i) == first_array_len for array_i in contents_list]
            if all(is_equal_length):
                can_convert = True
            else:
                can_convert = False
        else:
            can_convert = False
        
        return can_convert

    #
    #   Getter Methods
    #

    def __getitem__(self, key):
        if isinstance(key, self._arrays) or self._is_slice(key):
            item = self._get_with_array(key)
        else:
            item = super().__getitem__(key)
        return item
    
    def _get_with_array(self, array_key):
        item = {key: self._contents[key][array_key] for key in self.keys()}
        if self._type is list:
            item = list(item.values())
        return self.__class__(item)

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
        for key, val in key_container.items():
            if isinstance(val, self._arrays):
                self._set_array_item(key, val, new_value)
            else:
                self._contents[key][val] = new_value[key] if isinstance(new_value, Arraytainer) else new_value

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

    @property
    def shape(self):
        return self.get_shape()

    @property
    def ndim(self):
        return np.ndim(self)

    def reshape(self, *new_shapes, order='C'):

        new_shapes = list(new_shapes)

        for idx, shape in enumerate(new_shapes):
            if not isinstance(shape, (Arraytainer, *self._arrays)):
                shape = self._convert_to_array(shape)
                # Shape arrays must have at least one dimension for concatenate:
                if shape.ndim == 0:
                    shape = shape[None]
                new_shapes[idx] = shape
        new_shapes = np.concatenate(new_shapes)

        if len(new_shapes)==1 and isinstance(new_shapes[0], (Arraytainer, tuple)):
            new_shapes = new_shapes[0]

        return np.reshape(self, new_shapes, order=order)

    def flatten(self, order='C', return_array=True):
        output = np.ravel(self, order=order)
        if return_array:
            output = np.concatenate(output.list_elements())
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

        func_return = {}
        arraytainer_list = self._list_arraytainers_in_args(args) + self._list_arraytainers_in_args(kwargs)
        largest_arraytainer = self._find_largest_arraytainer(arraytainer_list)
        self._check_arraytainer_arg_compatability(arraytainer_list, largest_arraytainer)

        for key in largest_arraytainer.keys():
            args_i = self._prepare_func_args(args, key)
            kwargs_i = self._prepare_func_args(kwargs, key)
            func_return[key] = func(*args_i, **kwargs_i)

        if self._type is list:
            func_return = list(func_return.values())

        return self.__class__(func_return)

    @staticmethod
    def _list_arraytainers_in_args(args):
        args_iter = args.values() if isinstance(args, dict) else args
        args_arraytainers = []
        for arg_i in args_iter:
            if isinstance(arg_i, Arraytainer):
                args_arraytainers.append(arg_i)
            # Could have an arraytainer in a list in a tuple (e.g. np.concatenate):
            elif isinstance(arg_i, (list, tuple, dict)):
                # print('a')
                args_arraytainers += Arraytainer._list_arraytainers_in_args(arg_i)
                # print('b')
        
        return args_arraytainers

    @staticmethod
    def _find_largest_arraytainer(arraytainer_list):
        greatest_len = -1
        for arraytainer in arraytainer_list:
            if len(arraytainer) > greatest_len:
                greatest_len = len(arraytainer)
                largest_arraytainer = arraytainer
        return largest_arraytainer

    def _check_arraytainer_arg_compatability(self, arraytainer_list, largest_arraytainer):

        largest_keyset = set(largest_arraytainer.keys())
            
        for arraytainer_i in arraytainer_list:
            
            if arraytainer_i._type != largest_arraytainer._type:
                raise KeyError('Containers being combined through operations must',
                                'be all dictionary-like or all list-like')
            
            ith_keyset = set(arraytainer_i.keys())
            if not ith_keyset.issubset(largest_keyset):
                raise KeyError('Containers being combined through operations must',
                                'have identical sets of keys.')
    
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