import pytest
import numpy as np

@contextmanager
def no_error():
    yield

class ArraytainerTests(IndexingTests, SettingTests, MethodTests, FunctionTests):
    pass

class MethodTests:

    def test_shape_methods(self, contents, exception):
        arraytainer = self.container_class(contents)

        shape_func = lambda array : array.shape
        expected, exception = apply_func_to_contents(contents, func=shape_func)

        assert arraytainer.shape == expected
        assert arraytainer.shape_container.unpacked == self.container_class(expected)

    def test_unpacking(self, contents, exception):
        arraytainer = self.container_class(contents)
        assert arraytainer.unpacked == contents

    def test_boolean_methods(self, contents):
        arraytainer = self.container_class(contents)
        array_list = get_arrays_from_contents(contents)

        expected_all = all([np.all(x) for x in array_list])
        expected_any = any([np.any(x) for x in array_list])
        
        assert_equal(arraytainer.all().unpacked, expected_all)
        assert_equal(arraytainer.any().unpacked, expected_any)

    def test_sum_method(self, contents, exception):
        
        arraytainer = self.container_class(contents)
        array_list = get_arrays_from_contents(contents)

        with exception:
            result = arraytainer.sum()

        for i, array_i in enumerate(array_list):
            if i==0:
                expected = array_i
            else:
                expected += array_i

        assert_equal(result.unpacked, expected)

    def test_iterables(self, contents, exception):
        
        arraytainer = self.container_class(contents)

        if contents is dict:
            values = tuple(contents.values())
            keys = tuple(contents.keys())
            items = tuple((key, val) for key, val in contents.items())
        else:
            values = tuple(contents)
            keys = tuple(range(len(contents)))
            items = tuple((i, val) for i, val in enumerate(contents))
        
        assert arraytainer.values() == values
        assert arraytainer.keys() == keys
        assert arraytainer.items() == items

class IndexingTests:

    def test_indexing_with_hash(self, contents_and_key):
        
        arraytainer = self.container_class(contents)

        try:
            expected = self.array(contents[key])
            exception = no_error()
        except Exception as error:
            expected = None
            exception = pytest.raises(error.__class__)
        
        with exception:
            assert arraytainer[key] == expected

    def test_indexing_with_array_or_slice(self, contents_and_key):
        arraytainer = self.container_class(contents)

        index_func = lambda contents : contents[array_or_slice]
        expected, exception = apply_func_to_contents(contents, func=index_func)

        with exception:
            assert arraytainer[array_or_slice] == self.container_class(expected)

    def test_indexing_with_container(self, contents_and_key):
        
        arraytainer = self.container_class(contents)
        arraytainer_key = self.container_class(key_contents)
        
        index_func = lambda contents, idx : contents[idx][key_contents[idx]]
        expected, exception = apply_func_to_contents(contents, func=index_func)
        
        with exception:
            assert arraytainer[arraytainer_key] == self.container_class(expected)

class SettingTests:

    def test_setting_with_hash(self, contents_key_and_value):
        arraytainer = self.container_class(contents)
        
        with exception:
            arraytainer[key] = new_val

        assert arraytainer[key] == self.array(new_val)

    def test_setting_with_array_or_slice(self, contents_key_and_value):
        
        arraytainer = self.container_class(contents)

        with exception:
            to_set = self.container_class(new_val) if val_is_container else new_val
            arraytainer[array_or_slice] = to_set

        def setter_func(contents, array_or_slice, new_val):
            contents_copy = contents.copy()
            contents_copy[array_or_slice] = new_val
            return contents_new
        def index_args(args, idx):
            array_or_slice, new_val = args
            new_val_idxed = new_val[idx] if val_is_container else new_val
            return (array_or_slice, new_val_idxed)

        expected = apply_func_to_contents(contents, args=(array_or_slice,new_val), func=setter_func, index_args=index_args)

        assert arraytainer[array_or_slice] == self.container_class(expected)

    def test_setting_with_container(self, contents_key_and_value):
        
        arraytainer = self.container_class(contents)
        arraytainer_key = self.container_class(key_contents)

        with exception:
            to_set = self.container_class(new_val) if val_is_container else new_val
            arraytainer[arraytainer_key] = to_set

        def setter_func(contents, key_contents, new_val):
            contents_copy = contents.copy()
            contents_copy[key_contents] = new_val
            return contents_new
        def index_args(args, idx):
            key_contents, new_val = args
            new_val_idxed = new_val[idx] if val_is_container else new_val
            return (key_contents[idx], new_val_idxed)

        expected = apply_func_to_contents(contents, args=(key_contents,new_val), func=setter_func, index_args=index_args)

        assert arraytainer[arraytainer_key] == self.container_class(expected)

@pytest.mark.parametrize('arg, func', (, np.cos),
                                      (,)

)
class FunctionTests:

    def test_apply_function(self, contents, func):
        
        arraytainers = [self.container_class(content_i) for content_i in contents]

        with exception:
            result = func(*arraytainers, *args, **kwargs)

        expected = apply_func_to_contents(*contents, func=func, args=args, kwargs=kwargs)

        assert assert_equal(result.unpacked, expected, approx_equal=True)
    
# Helper functions:

def assert_equal(contents_1, contents_2, approx_equal=False):
  
  def assert_func(contents_1, contents_2):
    if approx_equal:
      assert np.allclose(contents_1, contents_2)
    else:
      assert contents_1 == contents_2

  apply_func_to_contents(contents_1, contents_2, func=assert_func)

def apply_func_to_contents(*contents, func=None, args=(), kwargs=None, index_args=None, return_list=False)

    try:
        expected = call_func_recursive(*contents, func, args, kwargs, index_args, return_list)
        exception = no_error()
    except Expection as error:
        expected = None
        exception = pytest.raises(error.__class__)

    return (expected, exception)

def call_func_recursive(*contents, func, args, kwargs, index_args, return_list):
    
    first_item = contents[0]

    if kwargs is None:
      kwargs = {}

    if index_args is None:
        index_args = lambda args, idx : args

    if isinstance(first_item, dict):
        contents_i = {key: [item_i[key] for item_i in contents] for key in first_item.keys()}
        result = {key: apply_func_to_contents(*contents_i[key], func, index_args(args,key), kwargs, index_args, return_list) 
                  for key in first_item.keys()}
        return list(result.values()) if return_list else result

    elif isinstance(first_item, list):
        contents_i = [[item_i[idx] for item_i in contents] for idx in range(len(first_item))]
        return [apply_func_to_contents(*contents_i[idx], func, index_args(args,idx), kwargs, index_args, return_list)  
                for idx in range(len(first_item))]

    else:
        return func(*contents, *args, **kwargs)

def flatten_contents(contents, array_list=None):

    array_list = [] if array_list is None else array_list
    keys = range(len(contents)) if isinstance(contents, list) else contents.keys()

    for key in keys:
        if isinstance(contents[key], (dict, list)):
            array_list = flatten_contents(contents[key], array_list)
        else:
            array_list.append(contents[key])
    
    return array_list