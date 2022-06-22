
import pytest
from arraytainers import Arraytainer, Jaxtainer
import numpy as np
import jax.numpy as jnp

import helpers

reshape_tests_1 = [
    # Simple dict:
    {'contents': {'a': np.array([[1,2],[3,4],[5,6]]), 'b': np.array([6,5,4,3,2,1])}, 
     'shape': (2,3),
     'order': None,
     'expected': {'a': np.array([[1, 2, 3],[4, 5, 6]]), 'b': np.array([[6, 5, 4],[3, 2, 1]])} },
    {'contents': {'a': np.array([[1,2],[3,4],[5,6]]), 'b': np.array([6,5,4,3,2,1])}, 
     'shape': (2,3),
     'order': 'C',
     'expected': {'a': np.array([[1, 2, 3],[4, 5, 6]]), 'b': np.array([[6, 5, 4],[3, 2, 1]])} },
    {'contents': {'a': np.array([[1,2],[3,4],[5,6]]), 'b': np.array([6,5,4,3,2,1])}, 
     'shape': (2,3),
     'order': 'F',
     'expected': {'a': np.array([[1, 5, 4],[3, 2, 6]]), 'b': np.array([[6, 4, 2],[5, 3, 1]])} },

    # Simple list:
    {'contents': [np.array([[1,2],[3,4],[5,6]]), np.array([6,5,4,3,2,1])], 
     'shape': (2,3),
     'order': None,
     'expected': [np.array([[1, 2, 3],[4, 5, 6]]), np.array([[6, 5, 4],[3, 2, 1]])] },
    {'contents': [np.array([[1,2],[3,4],[5,6]]), np.array([6,5,4,3,2,1])], 
     'shape': (2,3),
     'order': 'C',
     'expected': [np.array([[1, 2, 3],[4, 5, 6]]), np.array([[6, 5, 4],[3, 2, 1]])] },
    {'contents': [np.array([[1,2],[3,4],[5,6]]), np.array([6,5,4,3,2,1])], 
     'shape': (2,3),
     'order': 'F',
     'expected': [np.array([[1, 5, 4],[3, 2, 6]]), np.array([[6, 4, 2],[5, 3, 1]])] },

     # Nested arraytainer:
    {'contents': {'a': {'c': [np.array([[1,2],[3,4],[5,6]])]}, 'b': [{'a':np.array([6,5,4,3,2,1])}]},
     'shape': (2,3),
     'order': None,
     'expected': {'a': {'c': [np.array([[1, 2, 3],[4, 5, 6]])]}, 'b': [{'a':np.array([[6, 5, 4],[3, 2, 1]])}]} },
    {'contents': {'a': {'c': [np.array([[1,2],[3,4],[5,6]])]}, 'b': [{'a':np.array([6,5,4,3,2,1])}]},
     'shape': (2,3),
     'order': 'C',
     'expected': {'a': {'c': [np.array([[1, 2, 3],[4, 5, 6]])]}, 'b': [{'a':np.array([[6, 5, 4],[3, 2, 1]])}]} },
    {'contents': {'a': {'c': [np.array([[1,2],[3,4],[5,6]])]}, 'b': [{'a':np.array([6,5,4,3,2,1])}]}, 
     'shape': (2,3),
     'order': 'F',
     'expected': {'a': {'c': [np.array([[1, 5, 4],[3, 2, 6]])]}, 'b': [{'a':np.array([[6, 4, 2],[5, 3, 1]])}]} }
]
@pytest.mark.parametrize("contents, shape, order, expected", [(test['contents'], test['shape'], test['order'], test['expected']) for test in reshape_tests_1])
def test_arraytainer_reshape_with_tuple_shape_and_string_order(contents, shape, order, expected):
    contents = helpers.deepcopy_contents(contents)
    expected = helpers.deepcopy_contents(expected)

    arraytainer = Arraytainer(contents)
    if order is None:
        result_1 = arraytainer.reshape(shape)
        result_2 = arraytainer.reshape(*shape)
    else:
        result_1 = arraytainer.reshape(shape, order=order)
        result_2 = arraytainer.reshape(*shape, order=order)

    helpers.assert_equal(result_1.unpack(), expected)
    helpers.assert_equal(result_1, Arraytainer(expected))
    helpers.assert_equal(result_2.unpack(), expected)
    helpers.assert_equal(result_2, Arraytainer(expected))

@pytest.mark.parametrize("contents, shape, order, expected", [(test['contents'], test['shape'], test['order'], test['expected']) for test in reshape_tests_1])
def test_jaxtainer_reshape_with_tuple_shape_and_string_order(contents, shape, order, expected):
    contents = helpers.deepcopy_contents(contents, has_jax_arrays=True)
    expected = helpers.deepcopy_contents(expected, has_jax_arrays=True)

    jaxtainer = Jaxtainer(contents)
    if order is None:
        result_1 = jaxtainer.reshape(shape)
        result_2 = jaxtainer.reshape(*shape)
    else:
        result_1 = jaxtainer.reshape(shape, order=order)
        result_2 = jaxtainer.reshape(*shape, order=order)

    helpers.assert_equal(result_1.unpack(), expected)
    helpers.assert_equal(result_1, Jaxtainer(expected))
    helpers.assert_equal(result_2.unpack(), expected)
    helpers.assert_equal(result_2, Jaxtainer(expected))

reshape_tests_2 = [
    # Simple dict:
    {'contents': {'a': np.array([[1,2],[3,4],[5,6]]), 'b': np.array([6,5,4,3,2,1])}, 
     'shape': ({'a': (1,2,3), 'b': (3,2)},),
     'order': None,
     'expected': {'a': np.array([[[1, 2, 3],[4, 5, 6]]]), 'b': np.array([[6, 5],[4, 3],[2, 1]])} },
    {'contents': {'a': np.array([[1,2],[3,4],[5,6]]), 'b': np.array([6,5,4,3,2,1])}, 
     'shape': ({'a': (1,2,3), 'b': (3,2)},),
     'order': 'C',
     'expected': {'a': np.array([[[1, 2, 3],[4, 5, 6]]]), 'b': np.array([[6, 5],[4, 3],[2, 1]])} },
    {'contents': {'a': np.array([[1,2],[3,4],[5,6]]), 'b': np.array([6,5,4,3,2,1])}, 
     'shape': ({'a': (1,2,3), 'b': (3,2)},),
     'order': 'F',
     'expected': {'a': np.array([[[1, 5, 4],[3, 2, 6]]]), 'b': np.array([[6, 3],[5, 2],[4, 1]])} },
    {'contents': {'a': np.array([[1,2],[3,4],[5,6]]), 'b': np.array([6,5,4,3,2,1])}, 
     'shape': (1, {'a': (2,3), 'b': (3,2)}),
     'order': 'F',
     'expected': {'a': np.array([[[1, 5, 4],[3, 2, 6]]]), 'b': np.array([[[6, 3],[5, 2],[4, 1]]])} },
    {'contents': {'a': np.array([[1,2],[3,4],[5,6]]), 'b': np.array([6,5,4,3,2,1])}, 
     'shape': ({'a': (2,3), 'b': (3,2)}, 1),
     'order': 'F',
     'expected': {'a': np.array([[[1], [5], [4]],[[3], [2], [6]]]), 'b': np.array([[[6], [3]],[[5], [2]],[[4], [1]]])} },
    {'contents': {'a': np.array([[1,2],[3,4],[5,6]]), 'b': np.array([6,5,4,3,2,1])}, 
     'shape': (1, {'a': (2,3), 'b': (3,2)}, 1),
     'order': 'F',
     'expected': {'a': np.array([[[[1], [5], [4]],[[3], [2], [6]]]]), 'b': np.array([[[[6], [3]],[[5], [2]],[[4], [1]]]])} },

    # Simple list:
    {'contents': [np.array([[1,2],[3,4],[5,6]]), np.array([6,5,4,3,2,1])],
     'shape': ([(1,2,3), (3,2)],),
     'order': 'F',
     'expected': [np.array([[[1, 5, 4],[3, 2, 6]]]), np.array([[6, 3],[5, 2],[4, 1]])] },
    {'contents': [np.array([[1,2],[3,4],[5,6]]),  np.array([6,5,4,3,2,1])], 
     'shape': (1, [(1,2,3), (3,2)]),
     'order': 'F',
     'expected': [np.array([[[[1, 5, 4],[3, 2, 6]]]]), np.array([[[6, 3],[5, 2],[4, 1]]])] },
    {'contents': [np.array([[1,2],[3,4],[5,6]]), np.array([6,5,4,3,2,1])],
     'shape': ([(1,2,3), (3,2)], 1),
     'order': 'F',
     'expected': [np.array([[[[1], [5], [4]],[[3], [2], [6]]]]), np.array([[[6], [3]],[[5], [2]],[[4], [1]]])] },
    {'contents': [np.array([[1,2],[3,4],[5,6]]), np.array([6,5,4,3,2,1])],
     'shape': (1, [(1,2,3), (3,2)], 1),
     'order': 'F',
     'expected': [np.array([[[[[1], [5], [4]],[[3], [2], [6]]]]]), np.array([[[[6], [3]],[[5], [2]],[[4], [1]]]])] },

    # Nested arraytainer with key broadcasting:
    {'contents': {'a': {'c': [np.array([[1,2],[3,4],[5,6]])]}, 'b': [{'a':np.array([6,5,4,3,2,1])}]},
     'shape': ({'a': {'c': [(1,2,3)]}, 'b': (3,2)},),
     'order': 'F',
     'expected': {'a': {'c': [np.array([[[1, 5, 4],[3, 2, 6]]])]}, 'b': [{'a':np.array([[6, 3],[5, 2],[4, 1]])}]} },
    {'contents': {'a': {'c': [np.array([[1,2],[3,4],[5,6]])]}, 'b': [{'a':np.array([6,5,4,3,2,1])}]},
     'shape': (1, {'a': {'c': [(1,2,3)]}, 'b': (3,2)}),
     'order': 'F',
     'expected': {'a': {'c': [np.array([[[[1, 5, 4],[3, 2, 6]]]])]}, 'b': [{'a':np.array([[[6, 3],[5, 2],[4, 1]]])}]} },
    {'contents': {'a': {'c': [np.array([[1,2],[3,4],[5,6]])]}, 'b': [{'a':np.array([6,5,4,3,2,1])}]},
     'shape': ({'a': {'c': [(1,2,3)]}, 'b': (3,2)}, 1),
     'order': 'F',
     'expected': {'a': {'c': [np.array([[[[1], [5], [4]],[[3], [2], [6]]]])]}, 'b': [{'a':np.array([[[6], [3]],[[5], [2]],[[4], [1]]])}]} },
    {'contents': {'a': {'c': [np.array([[1,2],[3,4],[5,6]])]}, 'b': [{'a':np.array([6,5,4,3,2,1])}]},
     'shape': (1, {'a': {'c': [(1,2,3)]}, 'b': (3,2)}, 1),
     'order': 'F',
     'expected': {'a': {'c': [np.array([[[[[1], [5], [4]],[[3], [2], [6]]]]])]}, 'b': [{'a':np.array([[[[6], [3]],[[5], [2]],[[4], [1]]]])}]} },
]
@pytest.mark.parametrize("contents, shape, order, expected", [(test['contents'], test['shape'], test['order'], test['expected']) for test in reshape_tests_2])
def test_arraytainer_reshape_with_arraytainer_shape_and_string_order(contents, shape, order, expected):
    contents = helpers.deepcopy_contents(contents)
    expected = helpers.deepcopy_contents(expected)

    arraytainer = Arraytainer(contents)
    shape_w_arrayconversion = [Arraytainer(val, dtype=int) if not isinstance(val, int) else val for val in shape]
    shape_wo_arrayconversion = [Arraytainer(val, array_conversions=False, dtype=int) if not isinstance(val, int) else val for val in shape]
    if order is None:
        result_1 = arraytainer.reshape(shape_w_arrayconversion)
        result_2 = arraytainer.reshape(*shape_w_arrayconversion)
        result_3 = arraytainer.reshape(shape_wo_arrayconversion)
        result_4 = arraytainer.reshape(*shape_wo_arrayconversion)
    else:
        result_1 = arraytainer.reshape(shape_w_arrayconversion, order=order)
        result_2 = arraytainer.reshape(*shape_w_arrayconversion, order=order)
        result_3 = arraytainer.reshape(shape_wo_arrayconversion, order=order)
        result_4 = arraytainer.reshape(*shape_wo_arrayconversion, order=order)

    for result in (result_1, result_2, result_3, result_4):
        helpers.assert_equal(result.unpack(), expected)
        helpers.assert_equal(result, Arraytainer(expected))


@pytest.mark.parametrize("contents, shape, order, expected", [(test['contents'], test['shape'], test['order'], test['expected']) for test in reshape_tests_2])
def test_jaxtainer_reshape_with_jaxtainer_shape_and_string_order(contents, shape, order, expected):
    contents = helpers.deepcopy_contents(contents, has_jax_arrays=True)
    expected = helpers.deepcopy_contents(expected, has_jax_arrays=True)

    jaxtainer = Jaxtainer(contents)
    shape_w_arrayconversion = [Jaxtainer(val, dtype=int) if not isinstance(val, int) else val for val in shape]
    shape_wo_arrayconversion = [Jaxtainer(val, array_conversions=False, dtype=int) if not isinstance(val, int) else val for val in shape]
    if order is None:
        result_1 = jaxtainer.reshape(shape_w_arrayconversion)
        result_2 = jaxtainer.reshape(*shape_w_arrayconversion)
        result_3 = jaxtainer.reshape(shape_wo_arrayconversion)
        result_4 = jaxtainer.reshape(*shape_wo_arrayconversion)
    else:
        result_1 = jaxtainer.reshape(shape_w_arrayconversion, order=order)
        result_2 = jaxtainer.reshape(*shape_w_arrayconversion, order=order)
        result_3 = jaxtainer.reshape(shape_wo_arrayconversion, order=order)
        result_4 = jaxtainer.reshape(*shape_wo_arrayconversion, order=order)

    for result in (result_1, result_2, result_3, result_4):
        helpers.assert_equal(result.unpack(), expected)
        helpers.assert_equal(result, Jaxtainer(expected))

reshape_tests_3 = [
    # Simple dict:
    {'contents': {'a': np.array([[1,2],[3,4],[5,6]]), 'b': np.array([6,5,4,3,2,1])}, 
     'shape': (2,3),
     'order': {'a': 'C', 'b': 'C'},
     'expected': {'a': np.array([[1, 2, 3],[4, 5, 6]]), 'b': np.array([[6, 5, 4],[3, 2, 1]])} },
    {'contents': {'a': np.array([[1,2],[3,4],[5,6]]), 'b': np.array([6,5,4,3,2,1])}, 
     'shape': (2,3),
     'order': {'a': 'C', 'b': 'F'},
     'expected': {'a': np.array([[1, 2, 3],[4, 5, 6]]), 'b': np.array([[6, 4, 2],[5, 3, 1]])} },
    {'contents': {'a': np.array([[1,2],[3,4],[5,6]]), 'b': np.array([6,5,4,3,2,1])}, 
     'shape': (2,3),
     'order': {'a': 'F', 'b': 'C'},
     'expected': {'a': np.array([[1, 5, 4],[3, 2, 6]]), 'b':  np.array([[6, 5, 4],[3, 2, 1]])} }, 
    {'contents': {'a': np.array([[1,2],[3,4],[5,6]]), 'b': np.array([6,5,4,3,2,1])}, 
     'shape': (2,3),
     'order': {'a': 'F', 'b': 'F'},
     'expected': {'a': np.array([[1, 5, 4],[3, 2, 6]]), 'b': np.array([[6, 4, 2],[5, 3, 1]])} }, 
    
    # Simple list:
    {'contents': [np.array([[1,2],[3,4],[5,6]]), np.array([6,5,4,3,2,1])], 
     'shape': (2,3),
     'order': ['C', 'F'],
     'expected': [np.array([[1, 2, 3],[4, 5, 6]]), np.array([[6, 4, 2],[5, 3, 1]])] },
    {'contents': [np.array([[1,2],[3,4],[5,6]]), np.array([6,5,4,3,2,1])], 
     'shape': (2,3),
     'order': ['F', 'C'],
     'expected': [np.array([[1, 5, 4],[3, 2, 6]]), np.array([[6, 5, 4],[3, 2, 1]])] },
    {'contents': [np.array([[1,2],[3,4],[5,6]]), np.array([6,5,4,3,2,1])], 
     'shape': (2,3),
     'order': ['F', 'F'],
     'expected': [np.array([[1, 5, 4],[3, 2, 6]]), np.array([[6, 4, 2],[5, 3, 1]])] },

    # Nested arraytainer - broadcast on one set of keys, don't on another:
    {'contents': {'a': {'c': [np.array([[1,2],[3,4],[5,6]])]}, 'b': [{'a':np.array([6,5,4,3,2,1])}]},
     'shape': (2,3),
     'order': {'a': 'C', 'b': [{'a': 'F'}]},
     'expected': {'a': {'c': [np.array([[1, 2, 3],[4, 5, 6]])]}, 'b': [{'a':np.array([[6, 4, 2],[5, 3, 1]])}]} },
    {'contents': {'a': {'c': [np.array([[1,2],[3,4],[5,6]])]}, 'b': [{'a':np.array([6,5,4,3,2,1])}]},
     'shape': (2,3),
     'order': {'a': 'F', 'b': [{'a': 'C'}]},
     'expected': {'a': {'c': [np.array([[1, 5, 4],[3, 2, 6]])]}, 'b': [{'a':np.array([[6, 5, 4],[3, 2, 1]])}]} },
    {'contents': {'a': {'c': [np.array([[1,2],[3,4],[5,6]])]}, 'b': [{'a':np.array([6,5,4,3,2,1])}]},
     'shape': (2,3),
     'order': {'a': 'F', 'b': [{'a': 'F'}]},
     'expected': {'a': {'c': [np.array([[1, 5, 4],[3, 2, 6]])]}, 'b': [{'a':np.array([[6, 4, 2],[5, 3, 1]])}]} },
]
@pytest.mark.parametrize("contents, shape, order, expected", [(test['contents'], test['shape'], test['order'], test['expected']) for test in reshape_tests_3])
def test_arraytainer_reshape_with_tuple_shape_and_arraytainer_order(contents, shape, order, expected):
    contents = helpers.deepcopy_contents(contents)
    expected = helpers.deepcopy_contents(expected)
    order = helpers.deepcopy_contents(order)

    arraytainer = Arraytainer(contents)
    order = Arraytainer(order, array_conversions=False)
    result_1 = arraytainer.reshape(shape, order=order)
    result_2 = arraytainer.reshape(*shape, order=order)

    helpers.assert_equal(result_1.unpack(), expected)
    helpers.assert_equal(result_1, Arraytainer(expected))
    helpers.assert_equal(result_2.unpack(), expected)
    helpers.assert_equal(result_2, Arraytainer(expected))

@pytest.mark.parametrize("contents, shape, order, expected", [(test['contents'], test['shape'], test['order'], test['expected']) for test in reshape_tests_3])
def test_jaxtainer_reshape_with_tuple_shape_and_jaxtainer_order(contents, shape, order, expected):
    contents = helpers.deepcopy_contents(contents, has_jax_arrays=True)
    expected = helpers.deepcopy_contents(expected, has_jax_arrays=True)

    jaxtainer = Jaxtainer(contents)
    order = Jaxtainer(order, array_conversions=False)
    result_1 = jaxtainer.reshape(shape, order=order)
    result_2 = jaxtainer.reshape(*shape, order=order)

    helpers.assert_equal(result_1.unpack(), expected)
    helpers.assert_equal(result_1, Jaxtainer(expected))
    helpers.assert_equal(result_2.unpack(), expected)
    helpers.assert_equal(result_2, Jaxtainer(expected))

reshape_tests_4 = [
    # Simple dict:
    {'contents': {'a': np.array([[1,2],[3,4],[5,6]]), 'b': np.array([6,5,4,3,2,1])}, 
     'shape': ({'a': (1,2,3), 'b': (3,2)},),
     'order': {'a': 'F', 'b': 'F'},
     'expected': {'a': np.array([[[1, 5, 4],[3, 2, 6]]]), 'b': np.array([[6, 3],[5, 2],[4, 1]])} }, 
    {'contents': {'a': np.array([[1,2],[3,4],[5,6]]), 'b': np.array([6,5,4,3,2,1])}, 
     'shape': (1, {'a': (1,2,3), 'b': (3,2)}, 1),
     'order': {'a': 'C', 'b': 'F'},
     'expected': {'a': np.array([[[[[1], [2], [3]],[[4], [5], [6]]]]]), 'b': np.array([[[[6], [3]],[[5], [2]],[[4], [1]]]])} },
    {'contents': {'a': np.array([[1,2],[3,4],[5,6]]), 'b': np.array([6,5,4,3,2,1])}, 
     'shape': (1, {'a': (1,2,3), 'b': (3,2)}, 1),
     'order': {'a': 'F', 'b': 'C'},
     'expected': {'a': np.array([[[[[1], [5], [4]],[[3], [2], [6]]]]]), 'b': np.array([[[[6], [5]],[[4], [3]],[[2], [1]]]])} }, 
    {'contents': {'a': np.array([[1,2],[3,4],[5,6]]), 'b': np.array([6,5,4,3,2,1])}, 
     'shape': (1, {'a': (1,2,3), 'b': (3,2)}, 1),
     'order': {'a': 'F', 'b': 'F'},
     'expected': {'a': np.array([[[[[1], [5], [4]],[[3], [2], [6]]]]]), 'b': np.array([[[[6], [3]],[[5], [2]],[[4], [1]]]])} }, 

    # Simple list:
    {'contents': [np.array([[1,2],[3,4],[5,6]]), np.array([6,5,4,3,2,1])], 
     'shape': ([(1,2,3), (3,2)],),
     'order': ['F', 'F'],
     'expected': [np.array([[[1, 5, 4],[3, 2, 6]]]), np.array([[6, 3],[5, 2],[4, 1]])] }, 
    {'contents': [np.array([[1,2],[3,4],[5,6]]), np.array([6,5,4,3,2,1])], 
     'shape': (1, [(1,2,3), (3,2)], 1),
     'order': ['C', 'F'],
     'expected': [np.array([[[[[1], [2], [3]],[[4], [5], [6]]]]]),  np.array([[[[6], [3]],[[5], [2]],[[4], [1]]]])] },
    {'contents': [np.array([[1,2],[3,4],[5,6]]), np.array([6,5,4,3,2,1])], 
     'shape': (1, [(1,2,3), (3,2)], 1),
     'order': ['F', 'C'],
     'expected': [np.array([[[[[1], [5], [4]],[[3], [2], [6]]]]]), np.array([[[[6], [5]],[[4], [3]],[[2], [1]]]])] }, 
    {'contents': [np.array([[1,2],[3,4],[5,6]]), np.array([6,5,4,3,2,1])], 
     'shape': (1, [(1,2,3), (3,2)], 1),
     'order': ['F', 'F'],
     'expected': [np.array([[[[[1], [5], [4]],[[3], [2], [6]]]]]), np.array([[[[6], [3]],[[5], [2]],[[4], [1]]]])] }, 

     # Nested arraytainer:
    {'contents': {'a': {'c': [np.array([[1,2],[3,4],[5,6]])]}, 'b': [{'a':np.array([6,5,4,3,2,1])}]}, 
     'shape': ({'a': {'c': [(1,2,3)]}, 'b': (3,2)},),
     'order': {'a': 'F', 'b': [{'a': 'F'}]},
     'expected': {'a': {'c': [np.array([[[1, 5, 4],[3, 2, 6]]])]}, 'b': [{'a':np.array([[6, 3],[5, 2],[4, 1]])}]} },
    {'contents': {'a': {'c': [np.array([[1,2],[3,4],[5,6]])]}, 'b': [{'a':np.array([6,5,4,3,2,1])}]}, 
     'shape': (1, {'a': {'c': [(1,2,3)]}, 'b': (3,2)}, 1),
     'order': {'a': 'C', 'b': [{'a': 'F'}]},
     'expected': {'a': {'c': [np.array([[[[[1], [2], [3]],[[4], [5], [6]]]]])]}, 'b': [{'a':np.array([[[[6], [3]],[[5], [2]],[[4], [1]]]])}]}},
    {'contents': {'a': {'c': [np.array([[1,2],[3,4],[5,6]])]}, 'b': [{'a':np.array([6,5,4,3,2,1])}]}, 
     'shape': (1, {'a': {'c': [(1,2,3)]}, 'b': (3,2)}, 1),
     'order': {'a': 'F', 'b': [{'a': 'C'}]},
     'expected': {'a': {'c': [np.array([[[[[1], [5], [4]],[[3], [2], [6]]]]])]}, 'b': [{'a':np.array([[[[6], [5]],[[4], [3]],[[2], [1]]]])}]} }, 
    {'contents': {'a': {'c': [np.array([[1,2],[3,4],[5,6]])]}, 'b': [{'a':np.array([6,5,4,3,2,1])}]}, 
     'shape': (1, {'a': {'c': [(1,2,3)]}, 'b': (3,2)}, 1),
     'order': {'a': 'F', 'b': [{'a': 'F'}]},
     'expected': {'a': {'c': [np.array([[[[[1], [5], [4]],[[3], [2], [6]]]]])]}, 'b': [{'a':np.array([[[[6], [3]],[[5], [2]],[[4], [1]]]])}]}}, 
]
@pytest.mark.parametrize("contents, shape, order, expected", [(test['contents'], test['shape'], test['order'], test['expected']) for test in reshape_tests_4])
def test_arraytainer_reshape_with_arraytainer_shape_and_arraytainer_order(contents, shape, order, expected):
    contents = helpers.deepcopy_contents(contents)
    expected = helpers.deepcopy_contents(expected)

    arraytainer = Arraytainer(contents)
    order = Arraytainer(order, array_conversions=False)
    shape_w_arrayconversion = [Arraytainer(val, dtype=int) if not isinstance(val, int) else val for val in shape]
    shape_wo_arrayconversion = [Arraytainer(val, array_conversions=False, dtype=int) if not isinstance(val, int) else val for val in shape]

    result_1 = arraytainer.reshape(shape_w_arrayconversion, order=order)
    result_2 = arraytainer.reshape(*shape_w_arrayconversion, order=order)
    result_3 = arraytainer.reshape(shape_wo_arrayconversion, order=order)
    result_4 = arraytainer.reshape(*shape_wo_arrayconversion, order=order)

    for result in (result_1, result_2, result_3, result_4):
        helpers.assert_equal(result.unpack(), expected)
        helpers.assert_equal(result, Arraytainer(expected))

@pytest.mark.parametrize("contents, shape, order, expected", [(test['contents'], test['shape'], test['order'], test['expected']) for test in reshape_tests_4])
def test_jaxtainer_reshape_with_jaxtainer_shape_and_jaxtainer_order(contents, shape, order, expected):
    contents = helpers.deepcopy_contents(contents, has_jax_arrays=True)
    expected = helpers.deepcopy_contents(expected, has_jax_arrays=True)

    jaxtainer = Jaxtainer(contents)
    order = Jaxtainer(order, array_conversions=False)
    shape_w_arrayconversion = [Jaxtainer(val, dtype=int) if not isinstance(val, int) else val for val in shape]
    shape_wo_arrayconversion = [Jaxtainer(val, array_conversions=False, dtype=int) if not isinstance(val, int) else val for val in shape]

    result_1 = jaxtainer.reshape(shape_w_arrayconversion, order=order)
    result_2 = jaxtainer.reshape(*shape_w_arrayconversion, order=order)
    result_3 = jaxtainer.reshape(shape_wo_arrayconversion, order=order)
    result_4 = jaxtainer.reshape(*shape_wo_arrayconversion, order=order)

    for result in (result_1, result_2, result_3, result_4):
        helpers.assert_equal(result.unpack(), expected)
        helpers.assert_equal(result, Jaxtainer(expected))