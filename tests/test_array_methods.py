
import pytest
from arraytainers import Arraytainer, Jaxtainer
import numpy as np
import jax.numpy as jnp

import helpers

transpose_tests = [
    {'contents': {'a': np.array([[1,2],[3,4]]), 'b': np.array([[5,6],[7,8]])}, 
     'expected': {'a': np.array([[1,2],[3,4]]).T, 'b': np.array([[5,6],[7,8]]).T} },
    {'contents': [np.array([[1,2],[3,4]]), np.array([[5,6],[7,8]])], 
     'expected': [np.array([[1,2],[3,4]]).T, np.array([[5,6],[7,8]]).T] },
    {'contents': {'a': [np.array([[1,2],[3,4]]), np.array([[9],[10]])], 'b': {'c': [np.array([[5,6],[7,8]]), np.array([[[9,10]]])]}}, 
     'expected': {'a': [np.array([[1,2],[3,4]]).T, np.array([[9],[10]]).T], 'b': {'c': [np.array([[5,6],[7,8]]).T, np.array([[[9,10]]]).T]}} }
]
@pytest.mark.parametrize("contents, expected", [(test['contents'], test['expected']) for test in transpose_tests])
def test_arraytainer_transpose(contents, expected):
    contents = helpers.deepcopy_contents(contents)
    expected = helpers.deepcopy_contents(expected)
    arraytainer = Arraytainer(contents)
    
    result_1 = arraytainer.T
    result_2 = arraytainer.transpose()

    helpers.assert_equal(result_1.unpack(), expected)
    helpers.assert_equal(result_2.unpack(), expected)
    helpers.assert_equal(result_1, Arraytainer(expected))
    helpers.assert_equal(result_2, Arraytainer(expected))

@pytest.mark.parametrize("contents, expected", [(test['contents'], test['expected']) for test in transpose_tests])
def test_jaxtainer_transpose(contents, expected):
    contents = helpers.deepcopy_contents(contents, has_jax_arrays=True)
    expected = helpers.deepcopy_contents(expected, has_jax_arrays=True)
    arraytainer = Jaxtainer(contents)
    
    result_1 = arraytainer.T
    result_2 = arraytainer.transpose()

    helpers.assert_equal(result_1.unpack(), expected)
    helpers.assert_equal(result_2.unpack(), expected)
    helpers.assert_equal(result_1, Jaxtainer(expected))
    helpers.assert_equal(result_2, Jaxtainer(expected))

size_tests = [
    {'contents': {'a': np.ones((2,2)), 'b': np.ones((3,3,3)), 'c': 2}, 'expected': 2**2 + 3**3 + 1 },
    {'contents': [np.ones((2,2)), np.ones((3,3,3)), 2], 'expected': 2**2 + 3**3 + 1 },
    {'contents': {'a': [np.ones((3,3,3)), np.ones((2,2))], 'b': {'c': [np.ones((2,2,2)), 1]}}, 'expected': 3**3 + 2**2 + 2**3 + 1}
]
@pytest.mark.parametrize("contents, expected", [(test['contents'], test['expected']) for test in size_tests])
def test_arraytainer_size(contents, expected):
    contents = helpers.deepcopy_contents(contents)
    arraytainer = Arraytainer(contents)

    result = arraytainer.size

    assert result == expected
    assert type(result) == type(expected)

@pytest.mark.parametrize("contents, expected", [(test['contents'], test['expected']) for test in size_tests])
def test_jaxtainer_size(contents, expected):
    contents = helpers.deepcopy_contents(contents, has_jax_arrays=True)
    jaxtainer = Jaxtainer(contents)

    result = jaxtainer.size

    assert result == expected
    assert type(result) == type(expected)

flatten_tests = [
    # Simple dict:
    {'contents': {'a': np.array([[1,2,3],[4,5,6],[7,8,9]]), 'b': np.array([[6,5,4],[3,2,1]])},
     'order': None,
     'expected': np.array([1,2,3,4,5,6,7,8,9,6,5,4,3,2,1])},
    {'contents': {'a': np.array([[1,2,3],[4,5,6],[7,8,9]]), 'b': np.array([[6,5,4],[3,2,1]])},
     'order': 'C',
     'expected': np.array([1,2,3,4,5,6,7,8,9,6,5,4,3,2,1])},
    {'contents': {'a': np.array([[1,2,3],[4,5,6],[7,8,9]]), 'b': np.array([[6,5,4],[3,2,1]])},
     'order': 'F',
     'expected': np.array([1, 4, 7, 2, 5, 8, 3, 6, 9, 6, 3, 5, 2, 4, 1])},
    # Simple List:
    {'contents': [np.array([[1,2,3],[4,5,6],[7,8,9]]), np.array([[6,5,4],[3,2,1]])],
     'order': None,
     'expected': np.array([1,2,3,4,5,6,7,8,9,6,5,4,3,2,1])},
    {'contents': [np.array([[1,2,3],[4,5,6],[7,8,9]]), np.array([[6,5,4],[3,2,1]])],
     'order': 'C',
     'expected': np.array([1,2,3,4,5,6,7,8,9,6,5,4,3,2,1])},
    {'contents': [np.array([[1,2,3],[4,5,6],[7,8,9]]), np.array([[6,5,4],[3,2,1]])],
     'order': 'F',
     'expected': np.array([1, 4, 7, 2, 5, 8, 3, 6, 9, 6, 3, 5, 2, 4, 1])},
    # Nested arraytainers:
    {'contents': {'a': {'c': [np.array([[[1, 2],[3, 4]],[[5, 6],[7, 8]]]), np.array([[9,10]])]}, 1: [{'a': np.array([[11,12],[13,14]])}, np.array(15)]},
     'order': None,
     'expected': np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])},
    {'contents': {'a': {'c': [np.array([[[1, 2],[3, 4]],[[5, 6],[7, 8]]]), np.array([[9,10]])]}, 1: [{'a': np.array([[11,12],[13,14]])}, np.array(15)]},
     'order': 'C',
     'expected': np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])},
    {'contents': {'a': {'c': [np.array([[[1, 2],[3, 4]],[[5, 6],[7, 8]]]), np.array([[9,10]])]}, 1: [{'a': np.array([[11,12],[13,14]])}, np.array(15)]},
     'order': 'F',
     'expected': np.array([1, 5, 3, 7, 2, 6, 4, 8, 9, 10, 11, 13, 12, 14, 15])},
]
@pytest.mark.parametrize("contents, order, expected", [(test['contents'], test['order'], test['expected']) for test in flatten_tests])
def test_arraytainer_flatten_method(contents, order, expected):
    contents = helpers.deepcopy_contents(contents)

    arraytainer = Arraytainer(contents)
    if order is None:
        result = arraytainer.flatten()
    else:
        result = arraytainer.flatten(order=order)
    
    helpers.assert_equal(result, expected)

@pytest.mark.parametrize("contents, order, expected", [(test['contents'], test['order'], test['expected']) for test in flatten_tests])
def test_jaxtainer_flatten_method(contents, order, expected):
    contents = helpers.deepcopy_contents(contents, has_jax_arrays=True)

    jaxtainer = Jaxtainer(contents)
    if order is None:
        result = jaxtainer.flatten()
    else:
        result = jaxtainer.flatten(order=order)
    
    jax_expected = jnp.array(expected)
    helpers.assert_equal(result, jax_expected)
