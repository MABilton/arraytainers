import pytest
import numpy as np
import jax
import jax.numpy as jnp
from arraytainers import Arraytainer, Jaxtainer

import helpers

from_array_tests = [
    # Simple dict:
    {'array': np.arange(1,11),
     'shapes': {'a': (2,2), 'b': (3,2)},
     'order': None,
     'expected': {'a': np.array([[1,2],[3,4]]), 'b': np.array([[5,6],[7,8],[9,10]])}},
    {'array': np.arange(1,11),
     'shapes': {'a': (2,2), 'b': (3,2)},
     'order': 'C',
     'expected': {'a': np.array([[1,2],[3,4]]), 'b': np.array([[5,6],[7,8],[9,10]])}},
    {'array': np.arange(1,11),
     'shapes': {'a': (2,2), 'b': (3,2)},
     'order': 'F',
     'expected': {'a': np.array([[1,3],[2,4]]), 'b': np.array([[5,8],[6,9],[7,10]])}},
    {'array': np.array([[1,2,3],[4,5,6],[7,8,9]]),
     'shapes': {'a': (1,3), 'b': (3,2)},
     'order': 'C',
     'expected': {'a': np.array([[1,2,3]]), 'b': np.array([[4,5],[6,7],[8,9]])}}, 
    {'array': np.array([[1,2,3],[4,5,6],[7,8,9]]),
     'shapes': {'a': (1,3), 'b': (3,2)},
     'order': 'F',
     'expected': {'a': np.array([[1,2,3]]), 'b': np.array([[4,7],[5,8],[6,9]])}}, 

    # Simple list:
    {'array': np.arange(1,11),
     'shapes': [(2,2), (3,2)],
     'order': None,
     'expected': [np.array([[1,2],[3,4]]), np.array([[5,6],[7,8],[9,10]])]},
    {'array': np.arange(1,11),
     'shapes': [(2,2), (3,2)],
     'order': 'C',
     'expected': [np.array([[1,2],[3,4]]), np.array([[5,6],[7,8],[9,10]])]},
    {'array': np.arange(1,11),
     'shapes': [(2,2), (3,2)],
     'order': 'F',
     'expected': [np.array([[1,3],[2,4]]), np.array([[5,8],[6,9],[7,10]])] },
    {'array': np.array([[1,2,3],[4,5,6],[7,8,9]]),
     'shapes': [(1,3), (3,2)],
     'order': 'C',
     'expected': [np.array([[1,2,3]]), np.array([[4,5],[6,7],[8,9]])]}, 
    {'array': np.array([[1,2,3],[4,5,6],[7,8,9]]),
     'shapes': [(1,3), (3,2)],
     'order': 'F',
     'expected': [np.array([[1,2,3]]), np.array([[4,7],[5,8],[6,9]])] }, 

     # Nested contents:
    {'array': np.arange(1,15),
     'shapes': {'a': [{'c':(2,2)}, (2,2)], 'b': [(1,2), {'c': (2,2)}]},
     'order': None,
     'expected': {'a': [{'c': np.array([[1,2],[3,4]])}, np.array([[5,6],[7,8]])], 'b': [np.array([[9,10]]), {'c': np.array([[11,12],[13,14]])}]} },
    {'array': np.arange(1,15),
     'shapes': {'a': [{'c':(2,2)}, (2,2)], 'b': [(1,2), {'c': (2,2)}]},
     'order': 'C',
     'expected': {'a': [{'c': np.array([[1,2],[3,4]])}, np.array([[5,6],[7,8]])], 'b': [np.array([[9,10]]), {'c': np.array([[11,12],[13,14]])}]} },
    {'array': np.arange(1,15),
     'shapes': {'a': [{'c':(2,2)}, (2,2)], 'b': [(1,2), {'c': (2,2)}]},
     'order': 'F',
     'expected': {'a': [{'c': np.array([[1,3],[2,4]])}, np.array([[5,7],[6,8]])], 'b': [np.array([[9,10]]), {'c': np.array([[11,13],[12,14]])}]} },
    {'array': np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]]),
     'shapes': {'a': [{'c':(2,2)}, (2,2)], 'b': [(1,2), {'c': (2,2)}]},
     'order': 'C',
     'expected': {'a': [{'c': np.array([[1,2],[3,4]])}, np.array([[5,6],[7,8]])], 'b': [np.array([[9,10]]), {'c': np.array([[11,12],[13,14]])}]} }, 
    {'array': np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]]),
     'shapes': {'a': [{'c':(2,2)}, (2,2)], 'b': [(1,2), {'c': (2,2)}]},
     'order': 'F',
     'expected': {'a': [{'c': np.array([[1,3],[2,4]])}, np.array([[5,7],[6,8]])], 'b': [np.array([[9,10]]), {'c': np.array([[11,13],[12,14]])}]} } 
]
@pytest.mark.parametrize("array, shapes, order, expected", [(test['array'], test['shapes'], test['order'], test['expected']) for test in from_array_tests])
def test_arraytainer_construct_from_array(array, shapes, order, expected):
    array = helpers.deepcopy_contents(array)
    shapes = helpers.deepcopy_contents(shapes)
    expected = helpers.deepcopy_contents(expected)
    
    shapes = Arraytainer(shapes)
    if order is None:
        output = Arraytainer.from_array(array, shapes)
    else:
        output = Arraytainer.from_array(array, shapes, order=order)

    helpers.assert_equal(output, Arraytainer(expected))

@pytest.mark.parametrize("array, shapes, order, expected", [(test['array'], test['shapes'], test['order'], test['expected']) for test in from_array_tests])
def test_jaxtainer_construct_from_array(array, shapes, order, expected):
    array = helpers.deepcopy_contents(array, has_jax_arrays=True)
    shapes = helpers.deepcopy_contents(shapes, has_jax_arrays=True)
    expected = helpers.deepcopy_contents(expected, has_jax_arrays=True)
    
    shapes = Jaxtainer(shapes)
    if order is None:
        output = Jaxtainer.from_array(array, shapes)
    else:
        output = Jaxtainer.from_array(array, shapes, order=order)

    helpers.assert_equal(output, Jaxtainer(expected))