import numpy as np
import jax.numpy as jnp
import jaxlib
from copy import deepcopy
from itertools import product
from more_itertools import always_iterable
from arraytainers import Numpytainer, Jaxtainer

ARRAYTAINER_TYPES = (Numpytainer, Jaxtainer)
ARRAY_TYPES = (np.ndarray, jnp.DeviceArray, jaxlib.xla_extension.DeviceArray)
ARRAY_CONSTRUCTORS = (np.array, jnp.array)
NONARRAY_TYPES = (dict, list, tuple)
NUM_TYPES = (np.float32, np.float64)

def set_contents_item(contents, key_iterable, new_val):
    key_iterable = list(key_iterable)
    key_i = key_iterable.pop(0)
    if key_iterable:
        set_contents_item(contents[key_i], key_iterable, new_val)
    else:
        contents[key_i] = new_val

def get_contents_item(contents, key_iterable):
    key_iterable = list(key_iterable)
    key_i = key_iterable.pop(0)
    if key_iterable:
        return get_contents_item(contents[key_i], key_iterable)
    else:
        return contents[key_i]

def create_contents(shapes, array_constructor, single_value=None, return_bool=False, seed=42):
    shapes = deepcopy(shapes)
    np.random.seed(seed)
    in_contents = create_contents_recursion(shapes, array_constructor, single_value, return_bool)
    return in_contents

def create_contents_recursion(shapes, array_constructor, single_value, return_bool):

    if isinstance(shapes, dict):
        return {key: create_contents_recursion(val, array_constructor, single_value, return_bool) for key, val in shapes.items()}

    elif isinstance(shapes, list):
        return [create_contents_recursion(val, array_constructor, single_value, return_bool) for val in shapes]

    elif isinstance(shapes, tuple):
        if single_value is not None:
            array = single_value*np.ones(shapes)
            array = array.astype(bool) if return_bool else array
        else:
            array = np.random.rand(*shapes)
            array = array>0.5 if return_bool else 10*array
        return array_constructor(array)

# Functions to identify arrays and arraytainers:
def is_array(x):
    return isinstance(x, ARRAY_TYPES)

def is_arraytainer(x):
    return isinstance(x, ARRAYTAINER_TYPES)

# Funtion to create parameter combinations for test cases:
def cartesian_prod(*vals, flatten=True):
    input_vals = [val if isinstance(val,tuple) else (val,) for val in vals]
    prod = list(product(*input_vals))
    prod = [flatten_tuple(item) for item in prod] if flatten else prod
    return tuple(prod)

def flatten_tuple(input):
  flattened = []
  for vals in input:
    flattened += [val_i for val_i in vals] if isinstance(vals, tuple) else [vals]
  return flattened

def group_first_n_params(test_dict, n):
    return {key : [((*val[0:n],), *val[n:]) for val in cases] for key, cases in test_dict.items()}

# Helper functions to unpack parameter combos to paramterised test functions:
def unpack_test_cases(test_cases):
    return tuple([val_i for val in test_cases.values() for val_i in val])

def unpack_test_ids(test_cases):
    return tuple([key for key, val in test_cases.items() for _ in val])

# Helperfunction to get keys from dict or list/tuple:
def get_keys(contents):
    return contents.keys() if isinstance(contents, dict) else range(len(contents))

# Functions to assert that sets of unpacked contents are equal in typing at each level:
def assert_same_types(arraytainer_1, arraytainer_2, container_class=None, check_args=True):

    inputs_not_arraytainers = [not isinstance(x, ARRAYTAINER_TYPES) for x in (arraytainer_1, arraytainer_2)]
    if any(inputs_not_arraytainers):
        raise TypeError('Must pass arraytainers to assert_same_types.')

    try:
        assert_same_types_recursive(arraytainer_1, arraytainer_2, container_class=arraytainer_1.__class__, check_args=True)
    except Exception as e:
        raise Exception(f'Arraytainers {arraytainer_1} and {arraytainer_2} do not contain the same types')
        
def assert_same_types_recursive(arraytainer_1, arraytainer_2, container_class, check_args):

    # Check arraytainer types for first call of recursive function
    if check_args:
        assert type(arraytainer_1) == type(arraytainer_2)
    
    for key in arraytainer_1.keys():
        # It's possible that one value is a 32 bit number (if Jax array converted to Numpy) and the other 
        # is a 64 bit number - we'll consider all these types to be the same:
        arg_is_num = [isinstance(x, NUM_TYPES) for x in (arraytainer_1[key], arraytainer_2[key])]
        if any(arg_is_num):
            assert all(arg_is_num)
        else:
            assert type(arraytainer_1[key]) == type(arraytainer_2[key]) 
        # If arraytainer contains arraytainers, check typing of those entries too; if arraytainers don't have the same
        # set of keys, a KeyError will be thrown, which is dealt with elsewhere
        if isinstance(arraytainer_1[key], container_class):
            assert_same_types_recursive(arraytainer_1[key], arraytainer_2[key], container_class=container_class, check_args=False)

# Function to assert that to sets of unpacked contents are equal in value:
def assert_equal_values(contents_1, contents_2, approx_equal=True):
  
  inputs_are_arraytainers = [isinstance(x, (Numpytainer, Jaxtainer)) for x in (contents_1, contents_2)]

  if any(inputs_are_arraytainers):
      raise TypeError('Must pass unpacked version of arraytainer to assert_equal_values.')

  def assert_func(contents_1, contents_2):
    if approx_equal:
        # Need to adjust atol and rtol because, by default, Jax uses 32 bit numbers which means,
        # when compared to 64 bit Numpy computations, the default tolerances throw spurious errors:
      assert np.allclose(contents_1, contents_2, atol=1e-5, rtol=1e-5)
    else:
      assert contents_1 == contents_2

  try:
      # Must specify no key broadcasting - we want this to throw an error if we don't have matching sets of keys:
      apply_func_to_contents(contents_1, contents_2, key_broadcasting=False, func=assert_func, throw_exception=True)
  except Exception as e:
      print(contents_1)
      print(contents_2)
      raise Exception(f'Values {contents_1} and {contents_2} not equal.')

# Function to apply a function to a list of contents:
def apply_func_to_contents(*content_list, func=None, args=(), kwargs=None, index_args_fun=None, 
                            return_list=False, throw_exception=False, key_broadcasting=True):
  
    try:
        expected = call_func_recursive(content_list, func, args, kwargs, index_args_fun, key_broadcasting, return_list)
        exception = None
    except Exception as error:
        # Want to throw exception whe using call_func_recursive to assert that 
        # corresponding elements are equal in value/type:
        if throw_exception:
            raise error
        # Need to note which error has occurred if we wish to assert that a particular error occurs for a test case:
        else:
            expected = None
            exception = error

    return (expected, exception)

def call_func_recursive(content_list, func, args, kwargs, index_args_fun, key_broadcasting, return_list):

    # Initialise recursion variables for first call of call_func_recursive:
    kwargs = {} if kwargs is None else kwargs
    if index_args_fun is None:
      index_args_fun = lambda args, idx : args

    # Find non-Numpy/Jax array values in content_list:
    array_vals, nonarray_vals = [], []
    for val in content_list:
      nonarray_vals.append(val) if isinstance(val, NONARRAY_TYPES) else array_vals.append(val) 

    # If we're not performing key broadcasting, must have that keys of each item in content list are the same:
    if not key_broadcasting:
      throw_error = False
      # If not key broadcasting, all elements must be either arrays or non-arrays:
      if (len(nonarray_vals) < len(content_list)) and (len(array_vals) < len(content_list)):
        throw_error = True
      # Check to see if element keys are all same only if we have no array values 
      # - otherwise, we'll throw an unintended key error. At this point, if we
      # have arrays + nonarrays, the above condition should already cause an error to be thrown.
      if not array_vals:
        first_keys = set(get_keys(content_list[0]))
        for item in content_list[1:]:
          if set(get_keys(item)) != first_keys:
            throw_error = True
      if throw_error:
        raise KeyError('Incompatible keys passed to call_func_recursive.')

    if nonarray_vals:
        first_nonarray = nonarray_vals[0]
        # Assume that all nonarray values are of the same type; 
        # if they're not, an error will be thrown, which will be passed up to PyTest:
        if isinstance(first_nonarray, dict):
            # Group corresponding elements of each nonarray term in content_list -  Dictionary example:
            #       content_list == [{'a':element_1, 'b':element_2}, {'a':element_3, 'b':element_4}]
            # then corresponding_contents is a dictionary of the form:
            #       corresponding_contents = {'a':[element_1, element_3], 'b': [element_2, element_4]}
            # i.e. 'a' elements paired up and 'b' elements paired up.
            corresponding_contents = {key: [item_i[key] for item_i in nonarray_vals] for key in first_nonarray.keys()}
            # Also need to add 'lone' array elements:
            for key in first_nonarray.keys():
              corresponding_contents[key] += array_vals
            # Now that corresponding elements are paired together, recursively pass these onto the function along with 
            # 'lone' array elements we previously found:
            result = {}
            # Apply function to each corresponding set of contents, along with array lists:
            for key, contents_i in corresponding_contents.items():
                # Some args may also need to be indexed (e.g. keys when performing array indexing): 
                updated_args = index_args_fun(args,key)
                result[key] =  call_func_recursive(contents_i, func, updated_args, kwargs, index_args_fun, key_broadcasting, return_list) 
            return list(result.values()) if return_list else result
        elif isinstance(first_nonarray, (tuple,list)):
            # Group corresponding elements of each nonarray term in content_list -  List example:
            #       content_list == [[element_a, element_b], [element_c, element_d]]
            # then corresponding_contents is a list of the form:
            #       corresponding_contents = [[element_a, element_c], [element_b, element_d]]
            # # i.e. 0'th elements paired up and 1'st elements paired up.
            corresponding_contents = [[item_i[idx] for item_i in nonarray_vals] for idx in range(len(first_nonarray))]
            for idx, _ in enumerate(corresponding_contents):
              corresponding_contents[idx] += array_vals
            result = []
            for idx, contents_i in enumerate(corresponding_contents):
                updated_args = index_args_fun(args,idx)
                result.append(call_func_recursive(contents_i, func, updated_args, kwargs, index_args_fun, key_broadcasting, return_list) )
            return result
    else:
        return func(*content_list, *args, **kwargs)

# Array method utilities:
def sum_elements(contents):

    content_list = contents.values() if isinstance(contents,dict) else contents    
    array_elems, nonarray_elems = [], []
    for val in content_list:
        array_elems.append(val) if is_array(val) else nonarray_elems.append(val)
    
    if not nonarray_elems:
        sum_result = sum(array_elems)
    else:
        first_elem = nonarray_elems[0]
        elem_keys = get_keys(first_elem)
        sum_result = {key: sum_elements([val[key] for val in nonarray_elems] + array_elems) 
                     for key in elem_keys}
        if isinstance(first_elem, list):
            sum_result = list(sum_result.values())
    return sum_result

def get_list_of_arrays(contents, array_list=None):

    array_list = [] if array_list is None else array_list
    keys = get_keys(contents)

    for key in keys:
        if isinstance(contents[key], (dict, list)):
            array_list = get_list_of_arrays(contents[key], array_list)
        else:
            array_list.append(contents[key])
    
    return array_list