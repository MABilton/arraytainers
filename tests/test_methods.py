
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


# def test_flatten_method(contents, order, expected):
#     pass

# def test_copy_method():
#     pass

# def test_deepcopy_method():
#     pass
