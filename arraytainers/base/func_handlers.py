from copy import deepcopy

class Mixin:

    # Functions which deal with Numpy functions and universal operators:
    def __array_function__(self, func, types, args, kwargs):
        fun_return = self._manage_function_call(func, types, *args, **kwargs)
        return fun_return

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        fun_return = self._manage_function_call(ufunc, method, *args, **kwargs)
        return fun_return

    def _manage_function_call(self, func, types, args, kwargs):
        error_msg = ('The manage_function_call method is not implemented in',
                    'the base Arraytainer class. Instead, use either the',
                    'NumpyContainer or JaxContainer sub-classes.')
        raise AttributeError(' '.join(error_msg))

    # Helper functions used by manage_function_call in NumpyContainer and JaxContainer:
    def _prepare_args(self, args, key):
        args = [arg_i[key] if self.is_container(arg_i) else arg_i for arg_i in args]
        return args

    def _prepare_kwargs(self, kwargs, key):
        kwargs = {kwarg_key: tuple(val_i[key] if self.is_container(val_i) else val_i for val_i in val) 
                  for kwarg_key, val in kwargs.items()}
        return kwargs

    def _check_container_compatability(self, args, kwargs):
        
        arg_containers =  [arg_i for arg_i in args if self.is_container(arg_i)]
        kwarg_containers = [val_i for val_i in kwargs.values() if self.is_container(val_i)]
        container_list = arg_containers + kwarg_containers

        if container_list:
        # Get keys, type, and length of first container:
            container_0 = container_list[0]
            keys_0 = container_0.keys()
            type_0 = container_0._type
            len_0 = len(container_0)

        for container_i in container_list[1:]:
            # Ensure containers are either all dict-like or all list-like:
            if container_i._type != type_0:
                error_msg = ('Containers being combined through operations must',
                             'be all dictionary-like or all list-like')
                raise KeyError(' '.join(error_msg))

            # Ensure containers are all of same length:
            if len(container_i) != len_0:
                error_msg = ('Containers being combined through operations must',
                             'all contain the same number of elements.')
                raise KeyError(' '.join(error_msg))
            
            # Ensure containers have same keys:
            if container_i.keys() != keys_0:
                error_msg = ('Containers being combined through operations must',
                             'have identical sets of keys.')
                raise KeyError(' '.join(error_msg))