import pytest
import numpy as np
import jax.numpy as jnp
from itertools import product
from contextlib import contextmanager

@contextmanager
def no_error():
    yield

class ArraytainerTests:

    @pytest.mark.parametrize('contents, exception',
    [({(1,2): (1,)}, pytest.raises(KeyError)), 
    ({('1',2):(1,), (1,2):(1,)}, pytest.raises(KeyError)),
    ({('1',2):(1,), (1,'2'):(1,)}, None)], 
    indirect=['contents'])
    def test_key_checking(self, contents, exception):
        if exception is None:
            self.container_class(contents)
        else:
            with exception:
                self.container_class(contents)

    @pytest.mark.parametrize('contents_in, expected',
    [([[2]], [np.array([[2]])]), # Simple list
     ({'a': 2}, {'a': np.array(2)}), # Simple dict
     ([{'a': 2}, [[2,2],[2,2]]], [{'a': np.array(2)}, np.array([[2,2],[2,2]])]), # List with dict and list
     ([[np.array(2)]], [[np.array(2)]]), # List with array
     ({'a':[[2]],'b':{'c':[3]}}, {'a':np.array([[2]]),'b':{'c':np.array([3])}}), # Dict with array-convertible contents
     ({'a':[[np.array(2)]],'b':{'c':[3]}}, {'a':[[np.array(2)]],'b':{'c':np.array([3])}})]) 
    def test_array_conversion(self, contents_in, expected):
        arraytainer = self.container_class(contents_in)
        assert_equal(arraytainer.unpacked, expected)

    def test_shape_methods(self, std_contents):
        arraytainer = self.container_class(std_contents)

        shape_func = lambda array : array.shape
        expected, exception = apply_func_to_contents(std_contents, func=shape_func)

        assert_equal(arraytainer.shape, expected)
        assert_equal(arraytainer.shape_container.unpacked, expected)

    def test_unpacking(self, std_contents):
        arraytainer = self.container_class(std_contents)
        assert_equal(arraytainer.unpacked, std_contents)

    def test_iterables(self, std_contents):
        
        arraytainer = self.container_class(std_contents)

        if isinstance(std_contents, dict):
            values = tuple(std_contents.values())
            keys = tuple(std_contents.keys())
            items = tuple((key, val) for key, val in std_contents.items())            
        else:
            values = tuple(std_contents)
            keys = tuple(range(len(std_contents)))
            items = tuple((i, val) for i, val in enumerate(std_contents))
 
        assert set(arraytainer.keys()) == set(keys)
        assert_equal(arraytainer.values(unpacked=True), values)
        for result, expected in zip(arraytainer.items(unpacked=True), items):
            # Compare keys:
            assert result[0] == expected[0]
            # Compare arrays:
            assert_equal(result[1], expected[1])

    def test_boolean_methods(self, bool_contents):

        arraytainer = self.container_class(bool_contents)

        array_list = flatten_contents(bool_contents)

        assert arraytainer.all() == all([np.all(x) for x in array_list])
        assert arraytainer.any() == any([np.any(x) for x in array_list])

    @pytest.mark.parametrize('contents, exception', 
    [*product(([], {}), (None,)), # Empty
     *product(([(3,)], {'a': (3,)}, [[[(3,)]]], {'a':{'a':{'a':(3,)}}}, {'a':[{'a':(3,)}]}), (None,)), # Single values
     *product((3*[(3,3)], [(1,3), (3,3)], [[(3,3)], [(3,3)]], [(3,1), [(3,3), (3,3)]], 
              [[(3,1)],[[[(3,1)]]]]), (None,)), # Working lists
     *product(([(3,3),(3,2)], [[(3,2)],[(3,3), (3,3)]]), (Exception,)), # Broken lists 
     *product(({'a': (1,3) , 'b': (3,3)}, {'a': {'c':(3,3)}, 'b':{'c':(3,3)}}, {'a':(3,3),'b':{'c':(3,3)}}), (None,)), # Working dict 
     *product(({'a':(3,2),'b':(3,3)}, {'a':{'c':(3,3)},'b':{'d':(3,3)}}), (Exception,)),  # Broken dict 
     *product(([{'a':[(3,1)]}, {'a':[(3,1)]}], {'a':[(3,3), (3,3)], 'b':[(1,3), (3,3)]}, 
               {'b':[(1,3), {'c':(3,3)}]}, {'a':[(3,3), (3,3)], 'b':[[(3,3)], {'c':(3,3)}]}), (None,)), # Working mixed
     *product(([{'a':[(3,1), (3,1)]}, {'a':[(3,1)]}], [{'a':[(3,1)]}, {'b':[(3,1)]}], 
               {'a':[(3,3), (3,3)], 'b':[{'c':(3,3)}]}, 
               {'a':[(3,3)],'b':{'c':(3,3)}}), (Exception,)) # Broken mixed
    ], indirect=['contents'])
    def test_sum_elements_method(self, contents, exception):
        
        arraytainer = self.container_class(contents)

        if exception is None:
            expected = sum_elements(contents)
            result = arraytainer.sum_elements()
            # For comparisons, need to 'unpack' sum_result if it's an arraytainer:
            try:
                result = result.unpacked
            except AttributeError:
                pass
            assert_equal(result, expected)
        else:
            with pytest.raises((KeyError, IndexError, ValueError, TypeError)):
                arraytainer.sum_elements()

    @pytest.mark.parametrize('contents,exception', 
    [*product(([], {}), (None,)), # Empty
     *product(([(3,)], {'a': (3,)}, [[[(3,)]]], {'a':{'a':{'a':(3,)}}}, {'a':[{'a':(3,)}]}), (None,)), # Single values
     *product((3*[(3,3)], [(2,2), 2*[(2,2)]], [3*[(2,2)], 2*[(2,2)]]), (None,)), # Working list 
     *product(([(3,3), (2,1)], [[(2,2), (3,1)], 2*[(2,2)]]), (Exception,)), # Broken list 
     *product(({'a': (1,3) , 'b': (3,3)}, {'a': (3,3), 'b':{'c': (3,3)}}), (None,)), # Working dict 
     *product(({'a': (3,2) , 'b': (3,3)}, {'a': (3,3), 'b':{'c': (3,2)}}), (Exception,)),  # Broken dict 
     *product(([{'a':(1,3)}, [(3,3)]], {'a':{'c':[(3,3)]}, 'b':[(1,3), (3,3)]}), (None,)), # Working mixed
     *product(([{'a':(1,2)}, [(3,3)]], {'a':{'c':[(3,1)]}, 'b':[(1,2), (3,3)]}), (Exception,)) # Broken mixed
    ], indirect=['contents'])
    def test_sum_arrays_method(self, contents, exception):

        arraytainer = self.container_class(contents)
        
        if exception is None:
            array_list = flatten_contents(contents)
            expected = sum(array_list)
            assert np.allclose(arraytainer.sum_arrays(), expected)
        else:
            with pytest.raises((KeyError, IndexError, ValueError, TypeError)):
                arraytainer.sum_arrays()

    # def test_indexing_with_hash(self, contents, key, expected, expected_is_container):
        
    #     arraytainer = self.container_class(contents)

    #     index_func = lambda contents : contents[key]
    #     expected, exception = apply_func_to_contents(contents, func=index_func)
        
    #     with exception:
    #         assert arraytainer[key] == expected

    # def test_indexing_with_array_or_slice(self, arrays_contents_keys):
    #     arraytainer = self.container_class(contents)

    #     index_func = lambda contents : contents[array_or_slice]
    #     expected, exception = apply_func_to_contents(contents, func=index_func)

    #     with exception:
    #         assert arraytainer[array_or_slice] == self.container_class(expected)

    # def test_indexing_with_arraytainer(self, arraytainers_contents_keys):
        
    #     arraytainer = self.container_class(contents)
    #     arraytainer_key = self.container_class(key_contents)
        
    #     index_func = lambda contents, idx : contents[idx][key_contents[idx]]
    #     expected, exception = apply_func_to_contents(contents, func=index_func)
        
    #     with exception:
    #         assert arraytainer[arraytainer_key] == self.container_class(expected)

    # def test_set_with_hash(self, hash_contents_keys_values):
    #     arraytainer = self.container_class(contents)
        
    #     with exception:
    #         arraytainer[key] = new_val

    #     assert arraytainer[key] == self.array(new_val)

    # def test_set_nonarraytainer_with_array_or_slice(self, arrays_contents_keys_nonarraytainervalues):
        
    #     arraytainer = self.container_class(contents)

    #     expected, exception = apply_func_to_contents(contents, args=(array_or_slice, new_val), func=setter_func)

    #     with exception:
    #         arraytainer[array_or_slice] = new_val

    #     assert arraytainer[array_or_slice] == self.container_class(expected)

    # def test_set_nonarraytainer_with_arraytainer(self, arraytainers_contents_keys_nonarraytainervalues):
        
    #     arraytainer = self.container_class(contents)
    #     arraytainer_key = self.container_class(key_contents)

    #     with exception:
    #         to_set = self.container_class(new_val) if val_is_container else new_val
    #         arraytainer[arraytainer_key] = to_set

    #     def index_args(args, idx):
    #         key_contents, new_val = args
    #         new_val_idxed = new_val[idx] if val_is_container else new_val
    #         return (key_contents[idx], new_val_idxed)

    #     expected = apply_func_to_contents(contents, args=(key_contents,new_val), func=setter_func, index_args=index_args)

    #     assert arraytainer[arraytainer_key] == self.container_class(expected)

    # def test_set_arraytainer_with_array_or_slice(self, arrays_contents_keys_arraytainervalues):
        
    #     arraytainer = self.container_class(contents)
    #     val_arraytainer = self.container_class(val_contents)

    #     def index_args(args, idx):
    #         array_or_slice, new_val = args
    #         return (array_or_slice, new_val[idx])

    #     expected, exception = 
    #         apply_func_to_contents(contents, args=(array_or_slice, val_contents), func=setter_func, index_args=index_args)

    #     with exception:
    #         arraytainer[array_or_slice] = val_arraytainer

    #     assert arraytainer[array_or_slice] == self.container_class(expected)

    # def test_set_arraytainer_with_arraytainer(self, arraytainers_contents_keys_arraytainervalues):
        
    #     arraytainer = self.container_class(contents)
    #     key_arraytainer = self.container_class(key_contents)
    #     val_arraytainer = self.container_class(val_contents)

    #     def index_args(args, idx):
    #         key_contents, val_contents = args
    #         return (key_contents[idx], val_contents[idx])

    #     expected, exception = 
    #         apply_func_to_contents(contents, args=(key_contents, val_contents), func=setter_func, index_args=index_args)

    #     with exception:
    #         to_set = self.container_class(new_val) if val_is_container else new_val
    #         arraytainer[arraytainer_key] = to_set

    #     assert arraytainer[arraytainer_key] == self.container_class(expected)

    @pytest.mark.parametrize('single_arg_func', [np.exp, np.cos, np.floor, np.log, lambda x: x.T,
                                                 lambda x: x+2,  lambda x: x**2, lambda x: x*2, lambda x: x//2,
                                                 lambda x: 2-x,  lambda x: 2/x, lambda x: 2**x, lambda x: 2//x,
                                                 lambda x: 0.5*np.ones(1) @ x, lambda x: x @ 0.5*np.ones(1)])
    def test_apply_function_single_arg(self, std_contents, single_arg_func):
        
        arraytainer = self.container_class(std_contents)

        expected, exception = apply_func_to_contents(std_contents, func=single_arg_func)

        if exception is None:
            result = single_arg_func(arraytainer)
            assert_equal(result.unpacked, expected, approx_equal=True)
        else:
            with pytest.raises((KeyError, IndexError, ValueError, TypeError)):
                single_arg_func(arraytainer)


    # --- 

    
    # def test_apply_function_multiple_args(self, args_and_functions):
        
    #     arraytainers = [self.container_class(content_i) for content_i in arg_contents]

    #     with exception:
    #         result = func(*arraytainers)

    #     expected = apply_func_to_contents(*contents_list, func=func)

    #     assert assert_equal(result.unpacked, expected, approx_equal=True)

# Helper functions:
def setter_func(contents, key, new_val):
    contents_copy = contents.copy()
    contents_copy[key] = new_val
    return contents_new

def assert_equal(contents_1, contents_2, approx_equal=True):
  
  def assert_func(contents_1, contents_2):
    if approx_equal:
        # Need to adjust atol and rtol because, by default, Jax uses 32 bit numbers which means,
        # when compared to 64 bit Numpy computations, the default tolerances throw spurious errors:
      assert np.allclose(contents_1, contents_2, atol=1e-5, rtol=1e-5)
    else:
      assert contents_1 == contents_2

  apply_func_to_contents(contents_1, contents_2, func=assert_func, throw_exception=True)

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
            exception = pytest.raises(error.__class__)

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
    
    array_elems = {key:val for key, val in zip(keys,content_list) if isinstance(val, (np.ndarray, jnp.DeviceArray))}
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