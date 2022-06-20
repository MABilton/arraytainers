import pytest
from arraytainers import Arraytainer, Jaxtainer
import numpy as np
import jax.numpy as jnp

import helpers

simple_dict = {'a': -1*np.ones((2,2)), 'b': np.ones((2,2))}
dict_tests = [
 # Regular keys:
 { 'key': 'a', 'new_val': 2, 'expected': {'a': np.array(2), 'b': np.ones((2,2))} },
 { 'key': 1, 'new_val': 2, 'expected': {1: np.array(2), 'a': -1*np.ones((2,2)), 'b': np.ones((2,2))} },
 { 'key': 'b', 'new_val': np.ones((3,3,3)), 'expected': {'a': -1*np.ones((2,2)), 'b': np.ones((3,3,3))} },
 # Tuple keys:
 { 'key': (1,), 'new_val': 3, 'expected': {'a': np.array([[-1,-1],[3,3]]), 'b': np.array([[1,1],[3,3]])} },
 { 'key': (0,1), 'new_val': 3, 'expected': {'a': np.array([[-1,3],[-1,-1]]), 'b': np.array([[1,3],[1,1]])} },
 # Slice key:
 { 'key': slice(0,1), 'new_val': 3, 'expected': {'a': np.array([[3,3],[-1,-1]]), 'b': np.array([[3,3],[1,1]])} },
 { 'key': (slice(0,1), slice(0,1)), 'new_val': 3, 'expected': {'a': np.array([[3,-1],[-1,-1]]), 'b': np.array([[3,1],[1,1]])} },
 # Slice and tuple:
 { 'key': (slice(0,None), 1), 'new_val': 3, 'expected': {'a': np.array([[-1,3],[-1,3]]), 'b': np.array([[1,3],[1,3]])} },
 { 'key': (1, slice(None,1)), 'new_val': 3, 'expected': {'a': np.array([[-1,-1],[3,-1]]), 'b': np.array([[1,1],[3,1]])} },
 # Array keys:
 { 'key': np.array([True, False]), 'new_val': 2, 'expected': {'a': np.array([[2,2],[-1,-1]]), 'b': np.array([[2,2],[1,1]])} },
 { 'key': np.array([True, False]), 'new_val': np.array([3,4]), 'expected': {'a': np.array([[3,4],[-1,-1]]), 'b': np.array([[3,4],[1,1]])} },
 { 'key': np.array([[True, False], [True, False]]), 'new_val': 2, 'expected': {'a': np.array([[2,-1],[2,-1]]), 'b': np.array([[2,1],[2,1]])} },
 { 'key': np.array([[True, False], [True, False]]), 'new_val': np.array([3,4]), 'expected': {'a': np.array([[3,-1],[4,-1]]), 'b': np.array([[3,1],[4,1]])} },
 # Arraytainer keys:
 { 'key': Arraytainer({'a': np.array([True, False])}), 
   'new_val': 2, 
   'expected': {'a': np.array([[2,2],[-1,-1]]), 'b': np.ones((2,2))} },
 { 'key': Arraytainer({'a': np.array([True, False]), 'b': np.array([[True, False],[True, False]])}), 
   'new_val': 2, 
   'expected': {'a': np.array([[2,2],[-1,-1]]), 'b': np.array([[2,1],[2,1]])} },
 { 'key': Arraytainer({'a': np.array([[True, False],[True, False]]), 'b': np.array([True, False])}), 
   'new_val': np.array([3,4]),  
   'expected': {'a': np.array([[3,-1],[4,-1]]), 'b': np.array([[3,4],[1,1]])} },
 { 'key': Arraytainer({'a': np.array([[True, False],[True, False]]), 'b': np.array([True, False])}), 
   'new_val': Arraytainer({'a': 2, 'b': np.array([3,4])}),  
   'expected': {'a': np.array([[2,-1],[2,-1]]), 'b': np.array([[3,4],[1,1]])} }
]
@pytest.mark.parametrize("key, new_val, expected", [(test['key'], test['new_val'], test['expected']) for test in dict_tests])
def test_setting_with_simple_dict_arraytainer(key, new_val, expected):
    contents = helpers.deepcopy_contents(simple_dict, dtype=int)
    expected = helpers.convert_arrays_to_numpy(expected, dtype=int)
    new_val = helpers.convert_arrays_to_numpy(new_val, dtype=int)

    arraytainer = Arraytainer(contents)
    arraytainer[key] = new_val

    helpers.assert_equal(arraytainer.unpack(), expected)
    helpers.assert_equal(arraytainer, Arraytainer(expected))

jax_simple_dict = {'a': -1*jnp.ones((2,2)), 'b': jnp.ones((2,2))}
jax_dict_tests = [
 # Regular keys:
 { 'key': 'a', 'new_val': 2, 'expected': {'a': jnp.array(2), 'b': jnp.ones((2,2))} },
 { 'key': 1, 'new_val': 2, 'expected': {1: jnp.array(2), 'a': -1*jnp.ones((2,2)), 'b': jnp.ones((2,2))} },
 { 'key': 'b', 'new_val': jnp.ones((3,3,3)), 'expected': {'a': -1*jnp.ones((2,2)), 'b': jnp.ones((3,3,3))} },
 # Tuple keys:
 { 'key': (1,), 'new_val': 3, 'expected': {'a': jnp.array([[-1,-1],[3,3]]), 'b': jnp.array([[1,1],[3,3]])} },
 { 'key': (0,1), 'new_val': 3, 'expected': {'a': jnp.array([[-1,3],[-1,-1]]), 'b': jnp.array([[1,3],[1,1]])} },
 # Slice key:
 { 'key': slice(0,1), 'new_val': 3, 'expected': {'a': jnp.array([[3,3],[-1,-1]]), 'b': jnp.array([[3,3],[1,1]])} },
 { 'key': (slice(0,1), slice(0,1)), 'new_val': 3, 'expected': {'a': jnp.array([[3,-1],[-1,-1]]), 'b': jnp.array([[3,1],[1,1]])} },
 # Slice and tuple:
 { 'key': (slice(0,None), 1), 'new_val': 3, 'expected': {'a': jnp.array([[-1,3],[-1,3]]), 'b': jnp.array([[1,3],[1,3]])} },
 { 'key': (1, slice(None,1)), 'new_val': 3, 'expected': {'a': jnp.array([[-1,-1],[3,-1]]), 'b': jnp.array([[1,1],[3,1]])} },
 # Array keys:
 { 'key': jnp.array([True, False]), 'new_val': 2, 'expected': {'a': jnp.array([[2,2],[-1,-1]]), 'b': jnp.array([[2,2],[1,1]])} },
 { 'key': jnp.array([True, False]), 'new_val': jnp.array([3,4]), 'expected': {'a': jnp.array([[3,4],[-1,-1]]), 'b': jnp.array([[3,4],[1,1]])} },
 { 'key': jnp.array([[True, False], [True, False]]), 'new_val': 2, 'expected': {'a': jnp.array([[2,-1],[2,-1]]), 'b': jnp.array([[2,1],[2,1]])} },
 { 'key': jnp.array([[True, False], [True, False]]), 'new_val': jnp.array([3,4]), 'expected': {'a': jnp.array([[3,-1],[4,-1]]), 'b': jnp.array([[3,1],[4,1]])} },
 # Jaxtainer keys:
 { 'key': Jaxtainer({'a': jnp.array([True, False])}), 
   'new_val': 2, 
   'expected': {'a': jnp.array([[2,2],[-1,-1]]), 'b': jnp.ones((2,2))} },
 { 'key': Jaxtainer({'a': jnp.array([True, False]), 'b': jnp.array([[True, False],[True, False]])}), 
   'new_val': 2, 
   'expected': {'a': jnp.array([[2,2],[-1,-1]]), 'b': jnp.array([[2,1],[2,1]])} },
 { 'key': Jaxtainer({'a': jnp.array([[True, False],[True, False]]), 'b': jnp.array([True, False])}), 
   'new_val': jnp.array([3,4]),  
   'expected': {'a': jnp.array([[3,-1],[4,-1]]), 'b': jnp.array([[3,4],[1,1]])} },
 { 'key': Jaxtainer({'a': jnp.array([[True, False],[True, False]]), 'b': jnp.array([True, False])}), 
   'new_val': Jaxtainer({'a': 2, 'b': jnp.array([3,4])}),  
   'expected': {'a': jnp.array([[2,-1],[2,-1]]), 'b': jnp.array([[3,4],[1,1]])} }
]
@pytest.mark.parametrize("key, new_val, expected", [(test['key'], test['new_val'], test['expected']) for test in jax_dict_tests])
def test_setting_with_simple_dict_jaxtainer(key, new_val, expected):
    contents = helpers.deepcopy_contents(jax_simple_dict, has_jax_arrays=True, dtype=int)
    expected = helpers.convert_arrays_to_jax(expected, dtype=int)
    new_val = helpers.convert_arrays_to_jax(new_val, dtype=int)

    jaxtainer = Jaxtainer(contents)
    jaxtainer[key] = new_val

    helpers.assert_equal(jaxtainer.unpack(), expected)
    helpers.assert_equal(jaxtainer, Jaxtainer(expected))

simple_list = [-1*np.ones((2,2)), np.ones((2,2))]
list_tests = [
 # Regular keys: 
 { 'key': 0, 
   'new_val': 2, 
   'expected': [np.array(2), np.ones((2,2))] },
 { 'key': 1, 
   'new_val': np.ones((3,3,3)), 
   'expected': [-1*np.ones((2,2)), np.ones((3,3,3))] },
 # Tuple keys:
 { 'key': (1,), 
   'new_val': 3, 
   'expected': [np.array([[-1,-1],[3,3]]), np.array([[1,1],[3,3]])] },
 { 'key': (0,1), 
   'new_val': 3, 
   'expected': [np.array([[-1,3],[-1,-1]]), np.array([[1,3],[1,1]])] },
 # Slice key:
 { 'key': slice(0,1), 
   'new_val': 3, 
   'expected': [np.array([[3,3],[-1,-1]]), np.array([[3,3],[1,1]])] },
 { 'key': (slice(0,1), slice(0,1)), 
   'new_val': 3, 
   'expected': [np.array([[3,-1],[-1,-1]]), np.array([[3,1],[1,1]])] },
 # Slice and tuple:
 { 'key': (slice(0,None), 1), 
   'new_val': 3, 
   'expected': [np.array([[-1,3],[-1,3]]), np.array([[1,3],[1,3]])] },
 { 'key': (1, slice(None,1)), 
   'new_val': 3, 
   'expected': [np.array([[-1,-1],[3,-1]]), np.array([[1,1],[3,1]])] },
 # Array keys:
 { 'key': np.array([True, False]), 
   'new_val': 2, 
   'expected': [np.array([[2,2],[-1,-1]]), np.array([[2,2],[1,1]])] },
 { 'key': np.array([True, False]), 
   'new_val': np.array([3,4]), 
   'expected': [np.array([[3,4],[-1,-1]]), np.array([[3,4],[1,1]])] },
 { 'key': np.array([[True, False], [True, False]]), 
   'new_val': 2, 
   'expected': [np.array([[2,-1],[2,-1]]), np.array([[2,1],[2,1]])] },
 { 'key': np.array([[True, False], [True, False]]), 
   'new_val': np.array([3,4]), 
   'expected': [np.array([[3,-1],[4,-1]]), np.array([[3,1],[4,1]])] },
 # Arraytainer keys:
 { 'key': Arraytainer([np.array([True, False]), np.array([[True, False],[True, False]])]), 
   'new_val': 2, 
   'expected': [np.array([[2,2],[-1,-1]]), np.array([[2,1],[2,1]])] },
 { 'key': Arraytainer([np.array([[True, False],[True, False]]), np.array([True, False])]), 
   'new_val': np.array([3,4]),  
   'expected': [np.array([[3,-1],[4,-1]]), np.array([[3,4],[1,1]])] },
 { 'key': Arraytainer([np.array([[True, False],[True, False]]), np.array([True, False])]), 
   'new_val': Arraytainer([2, np.array([3,4])]),  
   'expected': [np.array([[2,-1],[2,-1]]), np.array([[3,4],[1,1]])] }
]
@pytest.mark.parametrize("key, new_val, expected", [(test['key'], test['new_val'], test['expected']) for test in list_tests])
def test_setting_with_simple_list_arraytainer(key, new_val, expected):
    contents = helpers.deepcopy_contents(simple_list, dtype=int)
    expected = helpers.convert_arrays_to_numpy(expected, dtype=int)
    new_val = helpers.convert_arrays_to_numpy(new_val, dtype=int)

    arraytainer = Arraytainer(contents)
    arraytainer[key] = new_val

    helpers.assert_equal(arraytainer.unpack(), expected)
    helpers.assert_equal(arraytainer, Arraytainer(expected))

jax_simple_list = [-1*jnp.ones((2,2)), jnp.ones((2,2))]
jax_list_tests = [
 # Regular keys: 
 { 'key': 0, 
   'new_val': 2, 
   'expected': [jnp.array(2), jnp.ones((2,2))] },
 { 'key': 1, 
   'new_val': jnp.ones((3,3,3)), 
   'expected': [-1*jnp.ones((2,2)), jnp.ones((3,3,3))] },
 # Tuple keys:
 { 'key': (1,), 
   'new_val': 3, 
   'expected': [jnp.array([[-1,-1],[3,3]]), jnp.array([[1,1],[3,3]])] },
 { 'key': (0,1), 
   'new_val': 3, 
   'expected': [jnp.array([[-1,3],[-1,-1]]), jnp.array([[1,3],[1,1]])] },
 # Slice key:
 { 'key': slice(0,1), 
   'new_val': 3, 
   'expected': [jnp.array([[3,3],[-1,-1]]), jnp.array([[3,3],[1,1]])] },
 { 'key': (slice(0,1), slice(0,1)), 
   'new_val': 3, 
   'expected': [jnp.array([[3,-1],[-1,-1]]), jnp.array([[3,1],[1,1]])] },
 # Slice and tuple:
 { 'key': (slice(0,None), 1), 
   'new_val': 3, 
   'expected': [jnp.array([[-1,3],[-1,3]]), jnp.array([[1,3],[1,3]])] },
 { 'key': (1, slice(None,1)), 
   'new_val': 3, 
   'expected': [jnp.array([[-1,-1],[3,-1]]), jnp.array([[1,1],[3,1]])] },
 # Array keys:
 { 'key': jnp.array([True, False]), 
   'new_val': 2, 
   'expected': [jnp.array([[2,2],[-1,-1]]), jnp.array([[2,2],[1,1]])] },
 { 'key': jnp.array([True, False]), 
   'new_val': jnp.array([3,4]), 
   'expected': [jnp.array([[3,4],[-1,-1]]), jnp.array([[3,4],[1,1]])] },
 { 'key': jnp.array([[True, False], [True, False]]), 
   'new_val': 2, 
   'expected': [jnp.array([[2,-1],[2,-1]]), jnp.array([[2,1],[2,1]])] },
 { 'key': jnp.array([[True, False], [True, False]]), 
   'new_val': jnp.array([3,4]), 
   'expected': [jnp.array([[3,-1],[4,-1]]), jnp.array([[3,1],[4,1]])] },
 # Jaxtainer keys:
 { 'key': Jaxtainer([jnp.array([True, False]), jnp.array([[True, False],[True, False]])]), 
   'new_val': 2, 
   'expected': [jnp.array([[2,2],[-1,-1]]), jnp.array([[2,1],[2,1]])] },
 { 'key': Jaxtainer([jnp.array([[True, False],[True, False]]), jnp.array([True, False])]), 
   'new_val': jnp.array([3,4]),  
   'expected': [jnp.array([[3,-1],[4,-1]]), jnp.array([[3,4],[1,1]])] },
 { 'key': Jaxtainer([jnp.array([[True, False],[True, False]]), jnp.array([True, False])]), 
   'new_val': Jaxtainer([2, jnp.array([3,4])]),  
   'expected': [jnp.array([[2,-1],[2,-1]]), jnp.array([[3,4],[1,1]])] }
]
@pytest.mark.parametrize("key, new_val, expected", [(test['key'], test['new_val'], test['expected']) for test in jax_list_tests])
def test_setting_with_simple_list_jaxtainer(key, new_val, expected):
    contents = helpers.deepcopy_contents(jax_simple_list, has_jax_arrays=True, dtype=int)
    expected = helpers.convert_arrays_to_jax(expected, dtype=int)
    new_val = helpers.convert_arrays_to_jax(new_val, dtype=int)

    jaxtainer = Jaxtainer(contents)
    jaxtainer[key] = new_val

    helpers.assert_equal(jaxtainer.unpack(), expected)
    helpers.assert_equal(jaxtainer, Jaxtainer(expected))

nested = {'a': [-1*np.ones((2,2)), 2*np.ones((2,2))], 'b': {'a': [np.ones((2,2))], 'b': np.ones((2,2))}}
nested_tests = [
  # Regular keys: 
  {'key': 'b', 
   'new_val': 2, 
   'expected': {'a': [-1*np.ones((2,2)), 2*np.ones((2,2))], 'b': np.array(2)} },
  {'key': 0,
   'new_val': 2, 
   'expected': {'a': [-1*np.ones((2,2)), 2*np.ones((2,2))], 'b': {'a': [np.ones((2,2))], 'b': np.ones((2,2))}, 0: np.array(2)} },
  # Tuple keys:
  { 'key': (1,), 
    'new_val': 3, 
    'expected': {'a': [np.array([[-1,-1],[3,3]]), np.array([[2,2],[3,3]])], 'b': {'a': [np.array([[1,1],[3,3]])], 'b': np.array([[1,1],[3,3]])}} },
  { 'key': (0,1), 
    'new_val': 3, 
    'expected': {'a': [np.array([[-1,3],[-1,-1]]), np.array([[2,3],[2,2]])], 'b': {'a': [np.array([[1,3],[1,1]])], 'b': np.array([[1,3],[1,1]])}} },
  # Slice key:
  { 'key': slice(0,1), 
    'new_val': 3, 
    'expected': {'a': [np.array([[3,3],[-1,-1]]), np.array([[3,3],[2,2]])], 'b': {'a': [np.array([[3,3],[1,1]])], 'b': np.array([[3,3],[1,1]])}} },
  { 'key': (slice(0,1), slice(0,1)), 
    'new_val': 3, 
    'expected': {'a': [np.array([[3,-1],[-1,-1]]), np.array([[3,2],[2,2]])], 'b': {'a': [np.array([[3,1],[1,1]])], 'b': np.array([[3,1],[1,1]])}} }, 
  # Slice and tuple:
  { 'key': (slice(0,None), 1), 
    'new_val': 3, 
    'expected': {'a': [np.array([[-1,3],[-1,3]]), np.array([[2,3],[2,3]])], 'b': {'a': [np.array([[1,3],[1,3]])], 'b': np.array([[1,3],[1,3]])}} },
  { 'key': (1, slice(None,1)), 
    'new_val': 3, 
    'expected': {'a': [np.array([[-1,-1],[3,-1]]), np.array([[2,2],[3,2]])], 'b': {'a': [np.array([[1,1],[3,1]])], 'b': np.array([[1,1],[3,1]])}} },
  # Array keys:
  { 'key': np.array([True, False]), 
    'new_val': 2, 
    'expected': {'a': [np.array([[2,2],[-1,-1]]), np.array([[2,2],[2,2]])], 'b': {'a': [np.array([[2,2],[1,1]])], 'b': np.array([[2,2],[1,1]])}} },
  { 'key': np.array([True, False]), 
    'new_val': np.array([3,4]), 
    'expected': {'a': [np.array([[3,4],[-1,-1]]), np.array([[3,4],[2,2]])], 'b': {'a': [np.array([[3,4],[1,1]])], 'b': np.array([[3,4],[1,1]])}} },
  { 'key': np.array([[True, False], [True, False]]), 
    'new_val': 3, 
    'expected': {'a': [np.array([[3,-1],[3,-1]]), np.array([[3,2],[3,2]])], 'b': {'a': [np.array([[3,1],[3,1]])], 'b': np.array([[3,1],[3,1]])}} },
  { 'key': np.array([[True, False], [True, False]]), 
    'new_val': np.array([3,4]), 
    'expected': {'a': [np.array([[3,-1],[4,-1]]), np.array([[3,2],[4,2]])], 'b': {'a': [np.array([[3,1],[4,1]])], 'b': np.array([[3,1],[4,1]])}} }, 
  # Arraytainer keys - no broadcasting:
  { 'key': Arraytainer({'b': {'a': [np.array([True, False])], 'b': np.array(0)}}), 
    'new_val': 3, 
    'expected': {'a': [-1*np.ones((2,2)), 2*np.ones((2,2))], 'b': {'a': [np.array([[3,3],[1,1]])], 'b': np.array([[3,3],[1,1]])}} },  
  { 'key': Arraytainer({'a': [np.array(1), np.array([[True, False],[False, True]])], 'b': {'a': [np.array([True, False])], 'b': np.array(0)}}), 
    'new_val': 3, 
    'expected': {'a': [np.array([[-1,-1],[3,3]]), np.array([[3,2],[2,3]])], 'b': {'a': [np.array([[3,3],[1,1]])], 'b': np.array([[3,3],[1,1]])}} },  
  { 'key': Arraytainer({'a': [np.array(1), np.array([[True, False],[False, True]])], 'b': {'a': [np.array([True, False])], 'b': np.array(0)}}), 
    'new_val': np.array([3,4]), 
    'expected': {'a': [np.array([[-1,-1],[3,4]]), np.array([[3,2],[2,4]])], 'b': {'a': [np.array([[3,4],[1,1]])], 'b': np.array([[3,4],[1,1]])}} }, 
  { 'key': Arraytainer({'a': [np.array(1), np.array([[True, False],[False, True]])], 'b': {'a': [np.array([True, False])], 'b': np.array(0)}}), 
    'new_val': Arraytainer({'a': [np.array([3,4]), np.array([5,6])], 'b': {'a': [np.array([7, 8])], 'b': np.array([9,10])}}), 
    'expected': {'a': [np.array([[-1,-1],[3,4]]), np.array([[5,2],[2,6]])], 'b': {'a': [np.array([[7,8],[1,1]])], 'b': np.array([[9,10],[1,1]])}} }, 
  # Arraytainer keys - broadcasting:
  { 'key': Arraytainer({'a': np.array([[True,False],[False,True]]), 'b': np.array([[False,True],[True,False]])}), 
    'new_val': 3, 
    'expected': {'a': [np.array([[3,-1],[-1,3]]), np.array([[3,2],[2,3]])], 'b': {'a': [np.array([[1,3],[3,1]])], 'b': np.array([[1,3],[3,1]])}} },
  { 'key': Arraytainer({'a': np.array([[True,False],[False,True]]), 'b': np.array([[False,True],[True,False]])}), 
    'new_val': np.array([3,4]), 
    'expected': {'a': [np.array([[3,-1],[-1,4]]), np.array([[3,2],[2,4]])], 'b': {'a': [np.array([[1,3],[4,1]])], 'b': np.array([[1,3],[4,1]])}} },
  { 'key': Arraytainer({'a': np.array([[True,False],[False,True]]), 'b': np.array([[False,True],[True,False]])}), 
    'new_val': Arraytainer({'a': np.array([3,4]), 'b': np.array([5,6])}), 
    'expected': {'a': [np.array([[3,-1],[-1,4]]), np.array([[3,2],[2,4]])], 'b': {'a': [np.array([[1,5],[6,1]])], 'b': np.array([[1,5],[6,1]])}} },
  { 'key': Arraytainer({'a': [np.array(1), np.array([[True, False],[False, True]])], 'b': {'a': [np.array([True, False])], 'b': np.array(0)}}), 
    'new_val': Arraytainer({'a': np.array([3,4]), 'b': np.array([5,6])}), 
    'expected': {'a': [np.array([[-1,-1],[3,4]]), np.array([[3,2],[2,4]])], 'b': {'a': [np.array([[5,6],[1,1]])], 'b': np.array([[5,6],[1,1]])}} },
  { 'key': Arraytainer({'a': np.array([[True,False],[False,True]]), 'b': np.array([[False,True],[True,False]])}), 
    'new_val': Arraytainer({'a': [np.array([3,4]), np.array([5,6])], 'b': {'a': [np.array([7, 8])], 'b': np.array([9,10])}}), 
    'expected': {'a': [np.array([[3,-1],[-1,4]]), np.array([[5,2],[2,6]])], 'b': {'a': [np.array([[1,7],[8,1]])], 'b': np.array([[1,9],[10,1]])}} }
]
@pytest.mark.parametrize("key, new_val, expected", [(test['key'], test['new_val'], test['expected']) for test in nested_tests])
def test_setting_with_nested_arraytainer(key, new_val, expected):
    
    contents = helpers.deepcopy_contents(nested, dtype=int)
    expected = helpers.convert_arrays_to_numpy(expected, dtype=int)
    new_val = helpers.convert_arrays_to_numpy(new_val, dtype=int)

    arraytainer = Arraytainer(contents)
    arraytainer[key] = new_val

    helpers.assert_equal(arraytainer.unpack(), expected)
    helpers.assert_equal(arraytainer, Arraytainer(expected))

jax_nested = {'a': [-1*jnp.ones((2,2)), 2*jnp.ones((2,2))], 'b': {'a': [jnp.ones((2,2))], 'b': jnp.ones((2,2))}}
jax_nested_tests = [
  # Regular keys: 
  {'key': 'b', 
   'new_val': 2, 
   'expected': {'a': [-1*jnp.ones((2,2)), 2*jnp.ones((2,2))], 'b': jnp.array(2)} },
  {'key': 0,
   'new_val': 2, 
   'expected': {'a': [-1*jnp.ones((2,2)), 2*jnp.ones((2,2))], 'b': {'a': [jnp.ones((2,2))], 'b': jnp.ones((2,2))}, 0: jnp.array(2)} },
  # Tuple keys:
  { 'key': (1,), 
    'new_val': 3, 
    'expected': {'a': [jnp.array([[-1,-1],[3,3]]), jnp.array([[2,2],[3,3]])], 'b': {'a': [jnp.array([[1,1],[3,3]])], 'b': jnp.array([[1,1],[3,3]])}} },
  { 'key': (0,1), 
    'new_val': 3, 
    'expected': {'a': [jnp.array([[-1,3],[-1,-1]]), jnp.array([[2,3],[2,2]])], 'b': {'a': [jnp.array([[1,3],[1,1]])], 'b': jnp.array([[1,3],[1,1]])}} },
  # Slice key:
  { 'key': slice(0,1), 
    'new_val': 3, 
    'expected': {'a': [jnp.array([[3,3],[-1,-1]]), jnp.array([[3,3],[2,2]])], 'b': {'a': [jnp.array([[3,3],[1,1]])], 'b': jnp.array([[3,3],[1,1]])}} },
  { 'key': (slice(0,1), slice(0,1)), 
    'new_val': 3, 
    'expected': {'a': [jnp.array([[3,-1],[-1,-1]]), jnp.array([[3,2],[2,2]])], 'b': {'a': [jnp.array([[3,1],[1,1]])], 'b': jnp.array([[3,1],[1,1]])}} }, 
  # Slice and tuple:
  { 'key': (slice(0,None), 1), 
    'new_val': 3, 
    'expected': {'a': [jnp.array([[-1,3],[-1,3]]), jnp.array([[2,3],[2,3]])], 'b': {'a': [jnp.array([[1,3],[1,3]])], 'b': jnp.array([[1,3],[1,3]])}} },
  { 'key': (1, slice(None,1)), 
    'new_val': 3, 
    'expected': {'a': [jnp.array([[-1,-1],[3,-1]]), jnp.array([[2,2],[3,2]])], 'b': {'a': [jnp.array([[1,1],[3,1]])], 'b': jnp.array([[1,1],[3,1]])}} },
  # Array keys:
  { 'key': jnp.array([True, False]), 
    'new_val': 2, 
    'expected': {'a': [jnp.array([[2,2],[-1,-1]]), jnp.array([[2,2],[2,2]])], 'b': {'a': [jnp.array([[2,2],[1,1]])], 'b': jnp.array([[2,2],[1,1]])}} },
  { 'key': jnp.array([True, False]), 
    'new_val': jnp.array([3,4]), 
    'expected': {'a': [jnp.array([[3,4],[-1,-1]]), jnp.array([[3,4],[2,2]])], 'b': {'a': [jnp.array([[3,4],[1,1]])], 'b': jnp.array([[3,4],[1,1]])}} },
  { 'key': jnp.array([[True, False], [True, False]]), 
    'new_val': 3, 
    'expected': {'a': [jnp.array([[3,-1],[3,-1]]), jnp.array([[3,2],[3,2]])], 'b': {'a': [jnp.array([[3,1],[3,1]])], 'b': jnp.array([[3,1],[3,1]])}} },
  { 'key': jnp.array([[True, False], [True, False]]), 
    'new_val': jnp.array([3,4]), 
    'expected': {'a': [jnp.array([[3,-1],[4,-1]]), jnp.array([[3,2],[4,2]])], 'b': {'a': [jnp.array([[3,1],[4,1]])], 'b': jnp.array([[3,1],[4,1]])}} }, 
  # Jaxtainer keys - no broadcasting:
  { 'key': Jaxtainer({'b': {'a': [jnp.array([True, False])], 'b': jnp.array(0)}}), 
    'new_val': 3, 
    'expected': {'a': [-1*jnp.ones((2,2)), 2*jnp.ones((2,2))], 'b': {'a': [jnp.array([[3,3],[1,1]])], 'b': jnp.array([[3,3],[1,1]])}} },  
  { 'key': Jaxtainer({'a': [jnp.array(1), jnp.array([[True, False],[False, True]])], 'b': {'a': [jnp.array([True, False])], 'b': jnp.array(0)}}), 
    'new_val': 3, 
    'expected': {'a': [jnp.array([[-1,-1],[3,3]]), jnp.array([[3,2],[2,3]])], 'b': {'a': [jnp.array([[3,3],[1,1]])], 'b': jnp.array([[3,3],[1,1]])}} },  
  { 'key': Jaxtainer({'a': [jnp.array(1), jnp.array([[True, False],[False, True]])], 'b': {'a': [jnp.array([True, False])], 'b': jnp.array(0)}}), 
    'new_val': jnp.array([3,4]), 
    'expected': {'a': [jnp.array([[-1,-1],[3,4]]), jnp.array([[3,2],[2,4]])], 'b': {'a': [jnp.array([[3,4],[1,1]])], 'b': jnp.array([[3,4],[1,1]])}} }, 
  { 'key': Jaxtainer({'a': [jnp.array(1), jnp.array([[True, False],[False, True]])], 'b': {'a': [jnp.array([True, False])], 'b': jnp.array(0)}}), 
    'new_val': Jaxtainer({'a': [jnp.array([3,4]), jnp.array([5,6])], 'b': {'a': [jnp.array([7, 8])], 'b': jnp.array([9,10])}}), 
    'expected': {'a': [jnp.array([[-1,-1],[3,4]]), jnp.array([[5,2],[2,6]])], 'b': {'a': [jnp.array([[7,8],[1,1]])], 'b': jnp.array([[9,10],[1,1]])}} }, 
  # Jaxtainer keys - broadcasting:
  { 'key': Jaxtainer({'a': jnp.array([[True,False],[False,True]]), 'b': jnp.array([[False,True],[True,False]])}), 
    'new_val': 3, 
    'expected': {'a': [jnp.array([[3,-1],[-1,3]]), jnp.array([[3,2],[2,3]])], 'b': {'a': [jnp.array([[1,3],[3,1]])], 'b': jnp.array([[1,3],[3,1]])}} },
  { 'key': Jaxtainer({'a': jnp.array([[True,False],[False,True]]), 'b': jnp.array([[False,True],[True,False]])}), 
    'new_val': jnp.array([3,4]), 
    'expected': {'a': [jnp.array([[3,-1],[-1,4]]), jnp.array([[3,2],[2,4]])], 'b': {'a': [jnp.array([[1,3],[4,1]])], 'b': jnp.array([[1,3],[4,1]])}} },
  { 'key': Jaxtainer({'a': jnp.array([[True,False],[False,True]]), 'b': jnp.array([[False,True],[True,False]])}), 
    'new_val': Jaxtainer({'a': jnp.array([3,4]), 'b': jnp.array([5,6])}), 
    'expected': {'a': [jnp.array([[3,-1],[-1,4]]), jnp.array([[3,2],[2,4]])], 'b': {'a': [jnp.array([[1,5],[6,1]])], 'b': jnp.array([[1,5],[6,1]])}} },
  { 'key': Jaxtainer({'a': [jnp.array(1), jnp.array([[True, False],[False, True]])], 'b': {'a': [jnp.array([True, False])], 'b': jnp.array(0)}}), 
    'new_val': Jaxtainer({'a': jnp.array([3,4]), 'b': jnp.array([5,6])}), 
    'expected': {'a': [jnp.array([[-1,-1],[3,4]]), jnp.array([[3,2],[2,4]])], 'b': {'a': [jnp.array([[5,6],[1,1]])], 'b': jnp.array([[5,6],[1,1]])}} },
  { 'key': Jaxtainer({'a': jnp.array([[True,False],[False,True]]), 'b': jnp.array([[False,True],[True,False]])}), 
    'new_val': Jaxtainer({'a': [jnp.array([3,4]), jnp.array([5,6])], 'b': {'a': [jnp.array([7, 8])], 'b': jnp.array([9,10])}}), 
    'expected': {'a': [jnp.array([[3,-1],[-1,4]]), jnp.array([[5,2],[2,6]])], 'b': {'a': [jnp.array([[1,7],[8,1]])], 'b': jnp.array([[1,9],[10,1]])}} }
]
@pytest.mark.parametrize("key, new_val, expected", [(test['key'], test['new_val'], test['expected']) for test in jax_nested_tests])
def test_setting_with_nested_jaxtainer(key, new_val, expected):
    
    contents = helpers.deepcopy_contents(jax_nested, has_jax_arrays=True, dtype=int)
    expected = helpers.convert_arrays_to_jax(expected, dtype=int)
    new_val = helpers.convert_arrays_to_jax(new_val, dtype=int)

    jaxtainer = Jaxtainer(contents)
    jaxtainer[key] = new_val

    helpers.assert_equal(jaxtainer.unpack(), expected)
    helpers.assert_equal(jaxtainer, Jaxtainer(expected))

index_errors = [
  {'contents': {'a': np.ones((2,2)), 'b': np.ones((3,3))},
   'key': np.array([True, False]),
   'new_val': 1},
  {'contents': [np.array(1), np.ones((2,2))],
   'key': np.array([True, False]),
   'new_val': 1}
]
@pytest.mark.parametrize("contents, key, new_val", [(test['contents'], test['key'], test['new_val']) for test in index_errors])
def test_arraytainer_setting_indexing_error(contents, key, new_val):
  arraytainer = Arraytainer(contents)
  with pytest.raises(IndexError):
    arraytainer[key] = new_val

broadcast_errors = [
  {'contents': {'a': np.ones((2,2)), 'b': np.ones((2,2))},
   'key': np.array([True, False]),
   'new_val': np.array([1,2,3])},
  {'contents': [np.ones((2,2)), np.ones((2,2))],
   'key': np.array([True, False]),
   'new_val': np.array([1,2,3])},
]
@pytest.mark.parametrize("contents, key, new_val", [(test['contents'], test['key'], test['new_val']) for test in broadcast_errors])
def test_arraytainer_setting_broadcast_error(contents, key, new_val):
  arraytainer = Arraytainer(contents)
  with pytest.raises(ValueError):
    arraytainer[key] = new_val

out_of_range_errors = [
  {'contents': [np.ones((2,2)), np.ones((2,2))],
   'key': 2,
   'new_val': np.array([1,2,3])},
  {'contents': [np.ones((2,2)), np.ones((2,2))],
   'key': 3,
   'new_val': Arraytainer([np.array([1,2,3])])}
]
@pytest.mark.parametrize("contents, key, new_val", [(test['contents'], test['key'], test['new_val']) for test in out_of_range_errors])
def test_arraytainer_setting_error_out_of_range_list_assignment(contents, key, new_val):
  arraytainer = Arraytainer(contents)
  with pytest.raises(KeyError, match='Unable to new assign items to a list-like Arraytainer'):
    arraytainer[key] = new_val

jax_index_errors = [
  {'contents': {'a': jnp.ones((2,2)), 'b': jnp.ones((3,3))},
   'key': jnp.array([True, False]),
   'new_val': 1},
  {'contents': [jnp.array(1), jnp.ones((2,2))],
   'key': jnp.array([True, False]),
   'new_val': 1}
]
@pytest.mark.parametrize("contents, key, new_val", [(test['contents'], test['key'], test['new_val']) for test in jax_index_errors])
def test_jaxtainer_setting_indexing_error(contents, key, new_val):
  jaxtainer = Jaxtainer(contents)
  with pytest.raises(IndexError):
    jaxtainer[key] = new_val

broadcast_errors = [
  {'contents': {'a': jnp.ones((2,2)), 'b': jnp.ones((2,2))},
   'key': jnp.array([True, False]),
   'new_val': jnp.array([1,2,3])},
  {'contents': [jnp.ones((2,2)), jnp.ones((2,2))],
   'key': jnp.array([True, False]),
   'new_val': jnp.array([1,2,3])},
]
@pytest.mark.parametrize("contents, key, new_val", [(test['contents'], test['key'], test['new_val']) for test in broadcast_errors])
def test_jaxtainer_setting_broadcast_error(contents, key, new_val):
  jaxtainer = Jaxtainer(contents)
  with pytest.raises(ValueError):
    jaxtainer[key] = new_val

out_of_range_errors = [
  {'contents': [jnp.ones((2,2)), jnp.ones((2,2))],
   'key': 2,
   'new_val': jnp.array([1,2,3])},
  {'contents': [jnp.ones((2,2)), jnp.ones((2,2))],
   'key': 3,
   'new_val': Jaxtainer([jnp.array([1,2,3])])}
]
@pytest.mark.parametrize("contents, key, new_val", [(test['contents'], test['key'], test['new_val']) for test in out_of_range_errors])
def test_jaxtainer_setting_error_out_of_range_list_assignment(contents, key, new_val):
  jaxtainer = Jaxtainer(contents)
  with pytest.raises(KeyError, match='Unable to new assign items to a list-like Arraytainer'):
    jaxtainer[key] = new_val
