
import numpy as np
import jax.numpy as jnp
import jaxlib

from itertools import product

ARRAY_TYPES = (np.ndarray, jnp.DeviceArray, jaxlib.xla_extension.DeviceArray)
NUM_TYPES = (np.float32, np.float64)

def create_idx_combos(contents, key_error_tuples):
    return [(contents, key, error) for contents, (key, error) in product((contents,), key_error_tuples)]

def assert_equal_values(contents_1, contents_2, approx_equal=True):
  
  def assert_func(contents_1, contents_2):
    if approx_equal:
        # Need to adjust atol and rtol because, by default, Jax uses 32 bit numbers which means,
        # when compared to 64 bit Numpy computations, the default tolerances throw spurious errors:
      assert np.allclose(contents_1, contents_2, atol=1e-5, rtol=1e-5)
    else:
      assert contents_1 == contents_2

  apply_func_to_contents(contents_1, contents_2, func=assert_func, throw_exception=True)

def assert_same_types(arraytainer_1, arraytainer_2, container_class=None, check_args=True):
    
    arg_is_array = [isinstance(x, ARRAY_TYPES) for x in (arraytainer_1, arraytainer_2)]

    if container_class is None:
        container_class = arraytainer_1.__class__

    if check_args:
        if any(arg_is_array):
            assert all(arg_is_array)
        else:
            assert type(arraytainer_1) == type(arraytainer_2)
    
    for key in arraytainer_1.keys():
        arg_is_array = [isinstance(x, ARRAY_TYPES) for x in (arraytainer_1[key], arraytainer_2[key])]
        arg_is_num = [isinstance(x, NUM_TYPES) for x in (arraytainer_1[key], arraytainer_2[key])]
        if any(arg_is_array):
            assert all(arg_is_array)
        elif any(arg_is_num):
            assert all(arg_is_num)
        else:
            assert type(arraytainer_1[key]) == type(arraytainer_2[key]) 
        if isinstance(arraytainer_1[key], container_class):
            assert_same_types(arraytainer_1[key], arraytainer_2[key], container_class=container_class, check_args=False)

def apply_func_to_contents(*contents, func=None, args=(), kwargs=None, index_args=None, 
                            return_list=False, throw_exception=False):

    try:
        expected = call_func_recursive(contents, func, args, kwargs, index_args, return_list)
        exception = None
    except Exception as error:
        if throw_exception:
            raise error
        else:
            expected = None
            exception = error

    return (expected, exception)

def call_func_recursive(contents, func, args, kwargs, index_args, return_list):
    
    first_item = contents[0]
    if kwargs is None:
      kwargs = {}

    if index_args is None:
        index_args = lambda args, idx : args

    if isinstance(first_item, dict):
        contents_i = {key: [item_i[key] for item_i in contents] for key in first_item.keys()}
        result = {key: call_func_recursive(contents_i[key], func, index_args(args,key), kwargs, index_args, return_list) 
                  for key in first_item.keys()}
        return list(result.values()) if return_list else result

    elif isinstance(first_item, (tuple,list)):
        contents_i = [[item_i[idx] for item_i in contents] for idx in range(len(first_item))]
        return [call_func_recursive(contents_i[idx], func, index_args(args,idx), kwargs, index_args, return_list)  
                for idx in range(len(first_item))]

    else:
        return func(*contents, *args, **kwargs)

def flatten_contents(contents, array_list=None):

    array_list = [] if array_list is None else array_list
    keys = get_keys(contents)

    for key in keys:
        if isinstance(contents[key], (dict, list)):
            array_list = flatten_contents(contents[key], array_list)
        else:
            array_list.append(contents[key])
    
    return array_list

def sum_elements(contents):

    keys = get_keys(contents)
    content_list = [contents[key] for key in keys]
    
    array_elems = {key:val for key, val in zip(keys,content_list) if isinstance(val, ARRAY_TYPES)}
    nonarray_elems = [val for key,val in zip(keys,content_list) if key not in array_elems.keys()]
    array_elems = array_elems.values()
    
    if not nonarray_elems:
        sum_result = sum(content_list)
    else:
        elem_keys = get_keys(nonarray_elems[0])
        sum_result = {key: sum_elements([*[val[key] for val in nonarray_elems], 
                                         *[val for val in array_elems]]) 
                     for key in elem_keys}
        if isinstance(nonarray_elems[0], list):
            sum_result = list(sum_result.values())
    return sum_result

def get_keys(contents):
    return contents.keys() if isinstance(contents, dict) else range(len(contents))