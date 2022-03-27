import warnings
import copy
import more_itertools

class Contents:

    _convert_to_contents = (tuple, list, dict)

    #
    #   Constructor Methods
    #

    def __new__(cls, *args, **kwargs):
        if cls is Contents:
            warnings.warn(f'Direct creation of {cls.__name__} class is not supported. '
                          'To avoid unexpected behaviour, create an Arraytainer or Jaxtainer object instead.')
        return object.__new__(cls)

    def __init__(self, contents, nested=True, **kwargs):
        self._contents = self._preprocess_contents(contents)
        if nested:
            self._nested_convert_to_contents(kwargs)

    def _preprocess_contents(self, contents):

        if self._get_contents_type(contents) is dict:
            self._check_keys(contents.keys())

        if isinstance(contents, Contents):
            contents = contents.unpack()
        elif not isinstance(contents, self._convert_to_contents):
            contents = [contents]

        if isinstance(contents, tuple):
            contents = list(contents)

        return self._deepcopy_contents(contents)

    def _nested_convert_to_contents(self, kwargs):
        for key, val in self.items():
            if isinstance(val, self._convert_to_contents):
                self._contents[key] = self.__class__(val, **kwargs)

    def _check_keys(self, keys):
        for key in more_itertools.always_iterable(keys):
            if isinstance(key, tuple):
                raise KeyError(f'The key {key} is a tuple, which are not allowed in {self.__class__.__name__}.')

    #
    #   Generic Methods
    #

    @staticmethod
    def _get_contents_type(val):
        return dict if hasattr(val, 'keys') else list

    @property
    def _type(self):
        return self._get_contents_type(self._contents)

    def __len__(self):
        return len(self._contents)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.unpack())})"

    #
    #   Copying Methods
    #

    # deepcopy changes jnp.array np.array, so Jaxtainer overloads this method:

    def __copy__(self):
        return self.__class__(self.unpack())

    def __deepcopy__(self, memo):
        return self.__class__(copy.deepcopy(self.unpack()))

    @staticmethod
    def _deepcopy_contents(contents):
        return copy.deepcopy(contents)

    def copy(self):
        return self.__copy__()

    def deepcopy(contents):
        return self.__deepcopy__()

    #
    #   Iterator Methods
    #

    def keys(self):
        if self._type is dict:
            keys = (key for key in self._contents.keys())
        else:
            keys = (i for i in range(len(self._contents)))
        return keys

    def values(self, unpacked=False):
        contents = self.unpack() if unpacked else self
        return (contents[key] for key in self.keys())

    def items(self, unpacked=False):
        contents = self.unpack() if unpacked else self
        return ((key, contents[key]) for key in self.keys())

    def __iter__(self):
        return iter(self._contents)
    
    #
    #   Getter Methods
    #

    def __getitem__(self, key, **kwargs):

        if isinstance(key, self._convert_to_contents):
            key = self.__class__(key)
    
        if isinstance(key, Contents):
            item = self._get_with_contents(key, kwargs)
        else:
            item = self._get_with_hash(key)
        
        return item
    
    def _get_with_contents(self, key_contents, kwargs):
        item = {key: self._contents[key][val] for key, val in key_contents.items()}
        item = list(item.values()) if self._type is list else item
        return self.__class__(item, **kwargs)   

    def _get_with_hash(self, key):
        try:
            item = self._contents[key]
        except KeyError:
            raise KeyError(f'{key} is not a key in this {self.__class__.__name__}; ',
                           f'valid keys for this {self.__class__.__name__} are {tuple(self.keys())}.')
        return item

    def unpack(self):
        unpacked = []
        for val in self.values():
            if isinstance(val, Contents):
                unpacked.append(val.unpack())
            else:
                unpacked.append(val)
        if self._type is dict:
            unpacked = dict(zip(self.keys(), unpacked))
        return unpacked
    
    @property
    def unpacked(self):
        return self.unpack()

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
            return self._contents[key_i]

    @property
    def contents(self):
        return self._contents

    def list_keys(self):
        return self._get_elements_and_keys(self.unpack(), return_elements=False, return_keys=True)

    def list_elements(self):
        return self._get_elements_and_keys(self.unpack(), return_elements=True, return_keys=False)

    def list_items(self):
        return self._get_elements_and_keys(self.unpack(), return_elements=True, return_keys=True)

    @staticmethod
    def _get_elements_and_keys(contents, return_elements, return_keys, output_list=None, key_list=None):
        
        if output_list is None:
            output_list = []
        
        if isinstance(contents, list):
            keys = range(len(contents))  
        else:
            keys = contents.keys()

        key_list_copy = copy.copy(key_list)
        for key in keys:

            if return_keys:
                if key_list_copy is None:
                    key_list = [key]
                else:
                    key_list = [*key_list_copy, key]

            if isinstance(contents[key], (dict,list)):
                output_list = \
                Contents._get_elements_and_keys(contents[key], return_elements, return_keys, output_list, key_list)
            else:

                if return_elements and return_keys:
                    output_list.append((tuple(key_list), contents[key]))
                elif return_keys:
                    output_list.append(tuple(key_list))
                elif return_elements:
                    output_list.append(contents[key])
        
        return output_list

    #
    #   Setter Methods
    #

    def __setitem__(self, key, new_value):
        
        self._check_keys(key)

        if isinstance(new_value, self._convert_to_contents):
            new_value = self.__class__(new_value)
        
        try:
            self._contents[key] = new_value
        except IndexError as error:
            if self._type is list:
                raise TypeError("Unable to new assign items to a list-like container; "
                                "use the update method instead.")
            raise error

    def update(self, new_val, *key_iterable):

        key_iterable = list(key_iterable)
        key = key_iterable.pop(0)
        if key_iterable:
            self[key].update(new_val, *key_iterable)
        else:
            if isinstance(self[key], Contents):
                if self._type is dict:
                    self[key]._contents.update(new_val)
                else:
                    self[key]._contents.append(new_val)
            else:
                self[key] = new_val

    #
    #   Function-Application Methods
    #

    # Applies a (potentially custom) functon to each value in contents:
    def apply(self, func, skip=0, broadcast=True, args=(), kwargs=None):
        
        if kwargs is None:
            kwargs = {}

        func_return = {}
        for key, val in self.items():
            if skip==0:
                func_return[key] = self._apply_func_to_contents(func, val, key, broadcast, args, kwargs)
            else:
                func_return[key] = val.apply(func, skip=skip-1, broadcast=broadcast, args=args, kwargs=kwargs) 

        if self._type is list:     
            func_return = list(func_return.values())

        return self.__class__(func_return)
    
    @staticmethod
    def _apply_func_to_contents(func, val, key, broadcast, args, kwargs):

        if broadcast:
            args_i = self._prepare_args_for_broadcasting(args, key)
            kwargs_i = self._prepare_apply_args_for_broadcasting(kwargs, key)
            if isinstance(val, Contents):
                output = val.apply(func, broadcast=True, args=args_i, kwargs=kwargs_i)
            else:
                output = func(val, *args_i, **kwargs_i)
        else:
            output = func(val, args, kwargs)

        return output 

    def _prepare_args_for_broadcasting(self, args, contents_key):
        
        args_iter = args.items() if isinstance(args, dict) else enumerate(args)

        prepped_args = {}
        for key, arg_i in args_iter:
            if isinstance(arg_i, Contents):
                prepped_args[key] = arg_i[contents_key]
            else:
                prepped_args[key] = arg_i

        if not isinstance(args, dict):
            prepped_args = tuple(prepped_args.values())

        return prepped_args