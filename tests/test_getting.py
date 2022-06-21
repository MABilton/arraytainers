import pytest
from arraytainers import Arraytainer, Jaxtainer
import numpy as np
import jax.numpy as jnp

import helpers

simple_dict = {'a': np.array([[1,2],[3,4]]), 'b': np.array([[5,6],[7,8]])}
dict_tests = [
 # Regular keys:
 { 'key': 'a', 'expected': np.array([[1,2],[3,4]]) },
 { 'key': 'b', 'expected': np.array([[5,6],[7,8]]) },
 # Tuple keys:
 { 'key': (1,), 'expected': {'a': np.array([3,4]), 'b': np.array([7,8])} },
 { 'key': (0,1), 'expected': {'a': np.array(2), 'b': np.array(6) } },
 # Slice key:
 { 'key': slice(0,1), 'expected': {'a': np.array([[1,2]]), 'b': np.array([[5,6]])} },
 { 'key': (slice(0,1), slice(0,1)), 'expected': {'a': np.array([[1]]), 'b': np.array([[5]])} },
 { 'key': (None, slice(None,None)), 'expected': {'a': np.array([[[1,2],[3,4]]]), 'b': np.array([[[5,6],[7,8]]])} },
 # Slice and tuple:
 { 'key': (slice(0,None), 1), 'expected': {'a': np.array([2,4]), 'b': np.array([6,8])} },
 { 'key': (1, slice(None,1)), 'expected': {'a': np.array([3]), 'b': np.array([7])} },
 # Array keys:
 { 'key': np.array([True, False]), 'expected': {'a': np.array([[1,2]]), 'b': np.array([[5,6]])} },
 { 'key': np.array([[True, False], [False, True]]), 'expected': {'a': np.array([1,4]), 'b': np.array([5,8])} },
 # Arraytainer keys:
 { 'key': Arraytainer({'a': np.array([True, False])}), 
   'expected': Arraytainer({'a': np.array([[1,2]])}) },
 { 'key': Arraytainer({'a': np.array([True, False]), 'b': np.array([[True, False],[False, True]])}), 
   'expected': Arraytainer({'a': np.array([[1,2]]), 'b': np.array([5,8])}) },
 { 'key': Arraytainer({'a': np.array([[True, False],[False, True]]), 'b': np.array([True, False])}),  
   'expected': Arraytainer({'a': np.array([1,4]), 'b': np.array([[5,6]])}) }
]
@pytest.mark.parametrize("key, expected", [(test['key'], test['expected']) for test in dict_tests])
def test_getting_with_simple_dict_arraytainer(key, expected):
    contents = helpers.deepcopy_contents(simple_dict, dtype=int)
    expected = helpers.convert_arrays_to_numpy(expected, dtype=int)

    arraytainer = Arraytainer(contents)
    gotten_val = arraytainer[key]

    helpers.assert_equal(gotten_val, expected)

jax_simple_dict = {'a': jnp.array([[1,2],[3,4]]), 'b': jnp.array([[5,6],[7,8]])}
jax_dict_tests = [
 # Regular keys:
 { 'key': 'a', 'expected': jnp.array([[1,2],[3,4]]) },
 { 'key': 'b', 'expected': jnp.array([[5,6],[7,8]]) },
 # Tuple keys:
 { 'key': (1,), 'expected': {'a': jnp.array([3,4]), 'b': jnp.array([7,8])} },
 { 'key': (0,1), 'expected': {'a': jnp.array(2), 'b': jnp.array(6) } },
 # Slice key:
 { 'key': slice(0,1), 'expected': {'a': jnp.array([[1,2]]), 'b': jnp.array([[5,6]])} },
 { 'key': (slice(0,1), slice(0,1)), 'expected': {'a': jnp.array([[1]]), 'b': jnp.array([[5]])} },
 { 'key': (None, slice(None,None)), 'expected': {'a': jnp.array([[[1,2],[3,4]]]), 'b': jnp.array([[[5,6],[7,8]]])} },
 # Slice and tuple:
 { 'key': (slice(0,None), 1), 'expected': {'a': jnp.array([2,4]), 'b': jnp.array([6,8])} },
 { 'key': (1, slice(None,1)), 'expected': {'a': jnp.array([3]), 'b': jnp.array([7])} },
 # Array keys:
 { 'key': jnp.array([True, False]), 'expected': {'a': jnp.array([[1,2]]), 'b': jnp.array([[5,6]])} },
 { 'key': jnp.array([[True, False], [False, True]]), 'expected': {'a': jnp.array([1,4]), 'b': jnp.array([5,8])} },
 # Jaxtainer keys:
 { 'key': Jaxtainer({'a': jnp.array([True, False])}), 
   'expected': Jaxtainer({'a': jnp.array([[1,2]])}) },
 { 'key': Jaxtainer({'a': jnp.array([True, False]), 'b': jnp.array([[True, False],[False, True]])}), 
   'expected': Jaxtainer({'a': jnp.array([[1,2]]), 'b': jnp.array([5,8])}) },
 { 'key': Jaxtainer({'a': jnp.array([[True, False],[False, True]]), 'b': jnp.array([True, False])}),  
   'expected': Jaxtainer({'a': jnp.array([1,4]), 'b': jnp.array([[5,6]])}) }
]
@pytest.mark.parametrize("key, expected", [(test['key'], test['expected']) for test in jax_dict_tests])
def test_getting_with_simple_dict_jaxtainer(key, expected):
    contents = helpers.deepcopy_contents(jax_simple_dict, has_jax_arrays=True, dtype=int)
    expected = helpers.convert_arrays_to_jax(expected, dtype=int)

    jaxtainer = Jaxtainer(contents)
    gotten_val = jaxtainer[key]

    helpers.assert_equal(gotten_val, expected)

simple_list = [np.array([[1,2],[3,4]]), np.array([[5,6],[7,8]])]
list_tests = [
 # Regular keys:
 { 'key': 0, 'expected': np.array([[1,2],[3,4]]) },
 { 'key': 1, 'expected': np.array([[5,6],[7,8]]) },
 # Tuple keys:
 { 'key': (1,), 'expected': [np.array([3,4]), np.array([7,8])] },
 { 'key': (0,1), 'expected': [np.array(2), np.array(6)] },
 # Slice key:
 { 'key': slice(0,1), 'expected': [np.array([[1,2]]), np.array([[5,6]])] },
 { 'key': (slice(0,1), slice(0,1)), 'expected': [np.array([[1]]), np.array([[5]])] },
 { 'key': (None, slice(None,None)), 'expected': [np.array([[[1,2],[3,4]]]), np.array([[[5,6],[7,8]]])] },
 # Slice and tuple:
 { 'key': (slice(0,None), 1), 'expected': [np.array([2,4]), np.array([6,8])] },
 { 'key': (1, slice(None,1)), 'expected': [np.array([3]), np.array([7])] },
 # Array keys:
 { 'key': np.array([True, False]), 'expected': [np.array([[1,2]]), np.array([[5,6]])] },
 { 'key': np.array([[True, False], [False, True]]), 'expected': [np.array([1,4]), np.array([5,8])] },
 # Arraytainer keys:
 { 'key': Arraytainer([np.array([True, False])]), 
   'expected': Arraytainer([np.array([[1,2]])]) },
 { 'key': Arraytainer([np.array([True, False]), np.array([[True, False],[False, True]])]), 
   'expected': Arraytainer([np.array([[1,2]]), np.array([5,8])]) },
 { 'key': Arraytainer([np.array([[True, False],[False, True]]), np.array([True, False])]),  
   'expected': Arraytainer([np.array([1,4]), np.array([[5,6]])]) }
]
@pytest.mark.parametrize("key, expected", [(test['key'], test['expected']) for test in list_tests])
def test_getting_with_simple_list_arraytainer(key, expected):
    contents = helpers.deepcopy_contents(simple_list, dtype=int)
    expected = helpers.convert_arrays_to_numpy(expected, dtype=int)

    arraytainer = Arraytainer(contents)
    gotten_val = arraytainer[key]

    helpers.assert_equal(gotten_val, expected)

jax_simple_list = [jnp.array([[1,2],[3,4]]), jnp.array([[5,6],[7,8]])]
jax_list_tests = [
 # Regular keys:
 { 'key': 0, 'expected': jnp.array([[1,2],[3,4]]) },
 { 'key': 1, 'expected': jnp.array([[5,6],[7,8]]) },
 # Tuple keys:
 { 'key': (1,), 'expected': [jnp.array([3,4]), jnp.array([7,8])] },
 { 'key': (0,1), 'expected': [jnp.array(2), jnp.array(6)] },
 # Slice key:
 { 'key': slice(0,1), 'expected': [jnp.array([[1,2]]), jnp.array([[5,6]])] },
 { 'key': (slice(0,1), slice(0,1)), 'expected': [jnp.array([[1]]), jnp.array([[5]])] },
 { 'key': (None, slice(None,None)), 'expected': [jnp.array([[[1,2],[3,4]]]), jnp.array([[[5,6],[7,8]]])] },
 # Slice and tuple:
 { 'key': (slice(0,None), 1), 'expected': [jnp.array([2,4]), jnp.array([6,8])] },
 { 'key': (1, slice(None,1)), 'expected': [jnp.array([3]), jnp.array([7])] },
 # Array keys:
 { 'key': jnp.array([True, False]), 'expected': [jnp.array([[1,2]]), jnp.array([[5,6]])] },
 { 'key': jnp.array([[True, False], [False, True]]), 'expected': [jnp.array([1,4]), jnp.array([5,8])] },
 # Jaxtainer keys:
 { 'key': Jaxtainer([jnp.array([True, False])]), 
   'expected': Jaxtainer([jnp.array([[1,2]])]) },
 { 'key': Jaxtainer([jnp.array([True, False]), jnp.array([[True, False],[False, True]])]), 
   'expected': Jaxtainer([jnp.array([[1,2]]), jnp.array([5,8])]) },
 { 'key': Jaxtainer([jnp.array([[True, False],[False, True]]), jnp.array([True, False])]),  
   'expected': Jaxtainer([jnp.array([1,4]), jnp.array([[5,6]])]) }
]
@pytest.mark.parametrize("key, expected", [(test['key'], test['expected']) for test in jax_list_tests])
def test_getting_with_simple_list_jaxtainer(key, expected):
    contents = helpers.deepcopy_contents(jax_simple_list, has_jax_arrays=True, dtype=int)
    expected = helpers.convert_arrays_to_jax(expected, dtype=int)

    jaxtainer = Jaxtainer(contents)
    gotten_val = jaxtainer[key]

    helpers.assert_equal(gotten_val, expected)

nested = {'a': [np.array([[1,2],[3,4]]), np.array([[5,6],[7,8]])], 'b': {'a': [np.array([[9,10],[11,12]])], 'b': np.array([[13,14],[15,16]])}}
nested_tests = [
  # Regular keys: 
  {'key': 'a',
   'expected': Arraytainer([np.array([[1,2],[3,4]]), np.array([[5,6],[7,8]])]) },
  {'key': 'b', 
   'expected': Arraytainer({'a': [np.array([[9,10],[11,12]])], 'b': np.array([[13,14],[15,16]])}) },
  # Tuple keys:
  { 'key': (1,), 
    'expected': Arraytainer({'a': [np.array([3,4]), np.array([7,8])], 'b': {'a': [np.array([11,12])], 'b': np.array([15,16])}}) },
  { 'key': (0,1), 
    'expected': Arraytainer({'a': [np.array(2), np.array(6)], 'b': {'a': [np.array(10)], 'b': np.array(14)}}) },
  # Slice key:
  { 'key': slice(0,1), 
    'expected': Arraytainer({'a': [np.array([[1,2]]), np.array([[5,6]])], 'b': {'a': [np.array([[9,10]])], 'b': np.array([[13,14]])}}) },
  { 'key': (slice(0,1), slice(0,1)), 
    'expected': Arraytainer({'a': [np.array([[1]]), np.array([[5]])], 'b': {'a': [np.array([[9]])], 'b': np.array([[13]])}}) }, 
  # Slice and tuple:
  { 'key': (slice(0,None), 1),
    'expected': Arraytainer({'a': [np.array([2,4]), np.array([6,8])], 'b': {'a': [np.array([10,12])], 'b': np.array([14,16])}}) },
  { 'key': (1, slice(None,1)), 
    'expected': Arraytainer({'a': [np.array([3]), np.array([7])], 'b': {'a': [np.array([11])], 'b': np.array([15])}}) },
  # Array keys:
  { 'key': np.array([True, False]), 
    'expected': Arraytainer({'a': [np.array([[1,2]]), np.array([[5,6]])], 'b': {'a': [np.array([[9,10]])], 'b': np.array([[13,14]])}}) },
  { 'key': np.array([[True, False], [False, True]]), 
    'expected': Arraytainer({'a': [np.array([1,4]), np.array([5,8])], 'b': {'a': [np.array([9,12])], 'b': np.array([13,16])}}) },
  { 'key': np.array([[True, False], [True, False]]), 
    'expected': Arraytainer({'a': [np.array([1,3]), np.array([5,7])], 'b': {'a': [np.array([9,11])], 'b': np.array([13,15])}})  }, 
  # Arraytainer keys - no broadcasting:
  { 'key': Arraytainer({'b': {'a': [np.array([True, False])], 'b': np.array(0)}}), 
    'expected': Arraytainer({'b': {'a': [np.array([[9,10]])], 'b': np.array([13,14])}}) },  
  { 'key': Arraytainer({'a': [np.array(1), np.array([[True, False],[False, True]])], 'b': {'a': [np.array([True, False])], 'b': np.array(0)}}), 
    'expected': Arraytainer({'a': [np.array([3,4]), np.array([5,8])], 'b': {'a': [np.array([[9,10]])], 'b': np.array([13,14])}}) },  
  # Arraytainer keys - broadcasting:
  { 'key': Arraytainer({'a': np.array([[True,False],[False,True]]), 'b': np.array([[False,True],[True,False]])}), 
    'expected': Arraytainer({'a': [np.array([1,4]), np.array([5,8])], 'b': {'a': [np.array([10,11])], 'b': np.array([14,15])}}) },
  { 'key': Arraytainer({'a': np.array([[True,False],[False,True]]), 'b': {'a': np.array([[False,True],[True,False]])}}), 
    'expected': Arraytainer({'a': [np.array([1,4]), np.array([5,8])], 'b': {'a': [np.array([10,11])]}}) },
]
@pytest.mark.parametrize("key, expected", [(test['key'], test['expected']) for test in nested_tests])
def test_getting_with_nested_arraytainer(key, expected):
    
    contents = helpers.deepcopy_contents(nested, dtype=int)
    expected = helpers.convert_arrays_to_numpy(expected, dtype=int)

    arraytainer = Arraytainer(contents)
    gotten_val = arraytainer[key]

    helpers.assert_equal(gotten_val, expected)

jax_nested = {'a': [jnp.array([[1,2],[3,4]]), jnp.array([[5,6],[7,8]])], 'b': {'a': [jnp.array([[9,10],[11,12]])], 'b': jnp.array([[13,14],[15,16]])}}
jax_nested_tests = [
  # Regular keys: 
  {'key': 'a',
   'expected': Jaxtainer([jnp.array([[1,2],[3,4]]), jnp.array([[5,6],[7,8]])]) },
  {'key': 'b', 
   'expected': Jaxtainer({'a': [jnp.array([[9,10],[11,12]])], 'b': jnp.array([[13,14],[15,16]])}) },
  # Tuple keys:
  { 'key': (1,), 
    'expected': Jaxtainer({'a': [jnp.array([3,4]), jnp.array([7,8])], 'b': {'a': [jnp.array([11,12])], 'b': jnp.array([15,16])}}) },
  { 'key': (0,1), 
    'expected': Jaxtainer({'a': [jnp.array(2), jnp.array(6)], 'b': {'a': [jnp.array(10)], 'b': jnp.array(14)}}) },
  # Slice key:
  { 'key': slice(0,1), 
    'expected': Jaxtainer({'a': [jnp.array([[1,2]]), jnp.array([[5,6]])], 'b': {'a': [jnp.array([[9,10]])], 'b': jnp.array([[13,14]])}}) },
  { 'key': (slice(0,1), slice(0,1)), 
    'expected': Jaxtainer({'a': [jnp.array([[1]]), jnp.array([[5]])], 'b': {'a': [jnp.array([[9]])], 'b': jnp.array([[13]])}}) }, 
  # Slice and tuple:
  { 'key': (slice(0,None), 1),
    'expected': Jaxtainer({'a': [jnp.array([2,4]), jnp.array([6,8])], 'b': {'a': [jnp.array([10,12])], 'b': jnp.array([14,16])}}) },
  { 'key': (1, slice(None,1)), 
    'expected': Jaxtainer({'a': [jnp.array([3]), jnp.array([7])], 'b': {'a': [jnp.array([11])], 'b': jnp.array([15])}}) },
  # Array keys:
  { 'key': jnp.array([True, False]), 
    'expected': Jaxtainer({'a': [jnp.array([[1,2]]), jnp.array([[5,6]])], 'b': {'a': [jnp.array([[9,10]])], 'b': jnp.array([[13,14]])}}) },
  { 'key': jnp.array([[True, False], [False, True]]), 
    'expected': Jaxtainer({'a': [jnp.array([1,4]), jnp.array([5,8])], 'b': {'a': [jnp.array([9,12])], 'b': jnp.array([13,16])}}) },
  { 'key': jnp.array([[True, False], [True, False]]), 
    'expected': Jaxtainer({'a': [jnp.array([1,3]), jnp.array([5,7])], 'b': {'a': [jnp.array([9,11])], 'b': jnp.array([13,15])}})  }, 
  # Jaxtainer keys - no broadcasting:
  { 'key': Jaxtainer({'b': {'a': [jnp.array([True, False])], 'b': jnp.array(0)}}), 
    'expected': Jaxtainer({'b': {'a': [jnp.array([[9,10]])], 'b': jnp.array([13,14])}}) },  
  { 'key': Jaxtainer({'a': [jnp.array(1), jnp.array([[True, False],[False, True]])], 'b': {'a': [jnp.array([True, False])], 'b': jnp.array(0)}}), 
    'expected': Jaxtainer({'a': [jnp.array([3,4]), jnp.array([5,8])], 'b': {'a': [jnp.array([[9,10]])], 'b': jnp.array([13,14])}}) },  
  # Jaxtainer keys - broadcasting:
  { 'key': Jaxtainer({'a': jnp.array([[True,False],[False,True]]), 'b': jnp.array([[False,True],[True,False]])}), 
    'expected': Jaxtainer({'a': [jnp.array([1,4]), jnp.array([5,8])], 'b': {'a': [jnp.array([10,11])], 'b': jnp.array([14,15])}}) },
  { 'key': Jaxtainer({'a': jnp.array([[True,False],[False,True]]), 'b': {'a': jnp.array([[False,True],[True,False]])}}), 
    'expected': Jaxtainer({'a': [jnp.array([1,4]), jnp.array([5,8])], 'b': {'a': [jnp.array([10,11])]}}) },
]
@pytest.mark.parametrize("key, expected", [(test['key'], test['expected']) for test in jax_nested_tests])
def test_getting_with_nested_jaxtainer(key, expected):
    
    contents = helpers.deepcopy_contents(jax_nested, has_jax_arrays=True, dtype=int)
    expected = helpers.convert_arrays_to_jax(expected, dtype=int)

    jaxtainer = Jaxtainer(contents)
    gotten_val = jaxtainer[key]

    helpers.assert_equal(gotten_val, expected)

successive_keys_contents = [{'a':1, 1:2}, {0:3, 1:4, 'a':{0:5}}, [6, 7, {'a':8, 'b':9}]]
successive_key_tests = [
    {'key_tuple': (0, 'a'), 'expected': np.array(1)},
    {'key_tuple': (1, 'a'), 'expected': Arraytainer({0:5})},
    {'key_tuple': (-1, 0), 'expected': np.array(6)},
    {'key_tuple': (-1, -1), 'expected': Arraytainer({'a':8, 'b':9})},
    {'key_tuple': (-1, -1, 'b'), 'expected': np.array(9)}
]
@pytest.mark.parametrize("key_tuple, expected", [(test['key_tuple'], test['expected']) for test in successive_key_tests])
def test_getting_with_nested_arraytainer_using_successive_regular_keys(key_tuple, expected):
    contents = helpers.deepcopy_contents(successive_keys_contents, dtype=int)
    expected = helpers.convert_arrays_to_numpy(expected, dtype=int)
    gotten_val = Arraytainer(contents)
    for key in key_tuple:
        gotten_val = gotten_val[key]
    helpers.assert_equal(gotten_val, expected)

jax_successive_keys_contents = [{'a':1, 1:2}, {0:3, 1:4, 'a':{0:5}}, [6, 7, {'a':8, 'b':9}]]
jax_successive_key_tests = [
    {'key_tuple': (0, 'a'), 'expected': jnp.array(1)},
    {'key_tuple': (1, 'a'), 'expected': Jaxtainer({0:5})},
    {'key_tuple': (-1, 0), 'expected': jnp.array(6)},
    {'key_tuple': (-1, -1), 'expected': Jaxtainer({'a':8, 'b':9})},
    {'key_tuple': (-1, -1, 'b'), 'expected': jnp.array(9)},
    {'key_tuple': (-1,-2), 'expected': jnp.array(7)}
]
@pytest.mark.parametrize("key_tuple, expected", [(test['key_tuple'], test['expected']) for test in jax_successive_key_tests])
def test_getting_with_nested_jaxtainer_using_successive_regular_keys(key_tuple, expected):
    contents = helpers.deepcopy_contents(jax_successive_keys_contents, dtype=int)
    expected = helpers.convert_arrays_to_jax(expected, dtype=int)
    gotten_val = Jaxtainer(contents)
    for key in key_tuple:
        gotten_val = gotten_val[key]
    helpers.assert_equal(gotten_val, expected)

key_error_contents = {'a': [0, 1], 'b': {'a':2, 'b':3}}
@pytest.mark.parametrize("key_tuple", [('c',), ('b',0), ('a',2), ('b',0), ('a','b')])
def test_arraytainer_getting_error_invalid_key(key_tuple):
    arraytainer = Arraytainer(key_error_contents)
    with pytest.raises(KeyError, match="not a key in this Arraytainer"):
        gotten_val = arraytainer
        for key in key_tuple:
            gotten_val = gotten_val[key]


jax_key_error_contents = {'a': [0, 1], 'b': {'a':2, 'b':3}}
@pytest.mark.parametrize("key_tuple", [('c',), ('b',0), ('a',2), ('b',0), ('a','b')])
def test_jaxtainer_getting_error_invalid_key(key_tuple):
    jaxtainer = Jaxtainer(jax_key_error_contents)
    with pytest.raises(KeyError, match="not a key in this Arraytainer"):
        gotten_val = jaxtainer
        for key in key_tuple:
            gotten_val = gotten_val[key]
