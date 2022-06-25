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

any_tests = [
    {'contents': {'a': [{'c': False}, np.zeros((2,2))], 'b': {'d':[np.array([False]), 0]}},
     'expected': False},
    {'contents': {'a': [{'c': True}, np.zeros((2,2))], 'b': {'d':[np.array([False]), 0]}},
     'expected': True},
    {'contents': {'a': [{'c': False}, np.array([[1,0],[0,0]])], 'b': {'d':[np.array([False]), 0]}},
     'expected': True},
    {'contents': {'a': [{'c': False}, np.zeros((2,2))], 'b': {'d':[np.array([False]), 1]}},
     'expected': True},
    {'contents': {'a': [{'c': False}, np.zeros((2,2))], 'b': {'d':[np.array([True]), 0]}},
     'expected': True}
]
@pytest.mark.parametrize("contents, expected", [(test['contents'], test['expected']) for test in any_tests])
def test_arraytainer_any_method(contents, expected):
    contents = helpers.deepcopy_contents(contents)

    arraytainer = Arraytainer(contents)
    result = arraytainer.any()
    
    assert isinstance(result, bool)
    assert result == expected

@pytest.mark.parametrize("contents, expected", [(test['contents'], test['expected']) for test in any_tests])
def test_jaxtainer_any_method(contents, expected):
    contents = helpers.deepcopy_contents(contents, has_jax_arrays=True)

    jaxtainer = Jaxtainer(contents)
    result = jaxtainer.any()
    
    assert isinstance(result, bool)
    assert result == expected

all_tests = [
    {'contents': {'a': [{'c': False}, np.zeros((2,2))], 'b': {'d':[np.array([False]), 0]}},
     'expected': False},
    {'contents': {'a': [{'c': True}, np.zeros((2,2))], 'b': {'d':[np.array([False]), 0]}},
     'expected': False},
    {'contents': {'a': [{'c': False}, np.array([[1,0],[0,0]])], 'b': {'d':[np.array([False]), 0]}},
     'expected': False},
    {'contents': {'a': [{'c': False}, np.zeros((2,2))], 'b': {'d':[np.array([False]), 1]}},
     'expected': False},
    {'contents': {'a': [{'c': False}, np.zeros((2,2))], 'b': {'d':[np.array([True]), 0]}},
     'expected': False},
    {'contents': {'a': [{'c': True}, np.ones((2,2))], 'b': {'d':[np.array([True]), 1]}},
     'expected': True}
]
@pytest.mark.parametrize("contents, expected", [(test['contents'], test['expected']) for test in all_tests])
def test_arraytainer_all_method(contents, expected):
    contents = helpers.deepcopy_contents(contents)

    arraytainer = Arraytainer(contents)
    result = arraytainer.all()
    
    assert isinstance(result, bool)
    assert result == expected

@pytest.mark.parametrize("contents, expected", [(test['contents'], test['expected']) for test in all_tests])
def test_jaxtainer_all_method(contents, expected):
    contents = helpers.deepcopy_contents(contents, has_jax_arrays=True)

    jaxtainer = Jaxtainer(contents)
    result = jaxtainer.all()
    
    assert isinstance(result, bool)
    assert result == expected

list_arrays_arraytainer_tests = [
    {'contents': {'a': 1, 'b': np.ones((3,3,3)), 'c': np.array([[1,2]])},
     'expected': [np.array(1), np.ones((3,3,3)), np.array([[1,2]])] },
    {'contents': [1, np.ones((3,3,3)), np.array([[1,2]])],
     'expected': [np.array(1), np.ones((3,3,3)), np.array([[1,2]])] }, 
    {'contents': {'a': {'c': [np.ones((3,3,3)), np.array([[9,10]])]}, 1: [{'a': np.array([[11,12],[13,14]])}, np.array(15)]},
     'expected': [np.ones((3,3,3)), np.array([[9,10]]), np.array([[11,12],[13,14]]), np.array(15)] },
    {'contents': {'a': [{'c': True}, np.ones((2,2))], 'b': {'d':[np.array([True]), 1]}},
     'expected': [np.array(True), np.ones((2,2)), np.array([True]), np.array(1)] }
]
@pytest.mark.parametrize("contents, expected", [(test['contents'], test['expected']) for test in list_arrays_arraytainer_tests])
def test_arraytainer_list_arrays_method(contents, expected):
    contents = helpers.deepcopy_contents(contents)

    arraytainer = Arraytainer(contents)
    result = arraytainer.list_arrays()
    
    assert isinstance(result, list)
    helpers.assert_equal(result, expected)

list_arrays_jaxtainer_tests = [
    {'contents': {'a': 1, 'b': jnp.ones((3,3,3)), 'c': jnp.array([[1,2]])},
     'expected': [jnp.array(1), jnp.ones((3,3,3)), jnp.array([[1,2]])] },
    {'contents': [1, jnp.ones((3,3,3)), jnp.array([[1,2]])],
     'expected': [jnp.array(1), jnp.ones((3,3,3)), jnp.array([[1,2]])] }, 
    {'contents': {'a': {'c': [jnp.ones((3,3,3)), jnp.array([[9,10]])]}, 1: [{'a': jnp.array([[11,12],[13,14]])}, jnp.array(15)]},
     'expected': [jnp.ones((3,3,3)), jnp.array([[9,10]]), jnp.array([[11,12],[13,14]]), jnp.array(15)] },
    {'contents': {'a': [{'c': True}, jnp.ones((2,2))], 'b': {'d':[jnp.array([True]), 1]}},
     'expected': [jnp.array(True), jnp.ones((2,2)), jnp.array([True]), jnp.array(1)] }
]
@pytest.mark.parametrize("contents, expected", [(test['contents'], test['expected']) for test in list_arrays_jaxtainer_tests])
def test_jaxtainer_list_arrays_method(contents, expected):
    contents = helpers.deepcopy_contents(contents, has_jax_arrays=True)

    jaxtainer = Jaxtainer(contents)
    result = jaxtainer.list_arrays()
    
    assert isinstance(result, list)
    helpers.assert_equal(result, expected)

@pytest.mark.parametrize("contents, expected", [(test['contents'], test['expected']) for test in list_arrays_arraytainer_tests])
def test_arraytainer_get_first_array_method(contents, expected):
    contents = helpers.deepcopy_contents(contents)

    arraytainer = Arraytainer(contents)
    result = arraytainer.first_array
    
    expected_first_array = expected[0]
    helpers.assert_equal(result, expected_first_array)

@pytest.mark.parametrize("contents, expected", [(test['contents'], test['expected']) for test in list_arrays_jaxtainer_tests])
def test_jaxtainer_get_first_array_method(contents, expected):
    contents = helpers.deepcopy_contents(contents, has_jax_arrays=True)

    jaxtainer = Jaxtainer(contents)
    result = jaxtainer.first_array
    
    expected_first_array = expected[0]
    helpers.assert_equal(result, expected_first_array)

sum_tests = [
    {'contents': {'a': np.array([[1,2,3]]), 'b': np.array([[4,5],[6,7]]), 'c': 8},
     'expected': sum(x for x in range(1,9))},
    {'contents': [np.array([[1,2,3]]), np.array([[4,5],[6,7]]), 8],
     'expected': sum(x for x in range(1,9))},
    {'contents': {'a': [np.array([[1,2,3]])], 'b': {'c': [np.array([[4,5],[6,7]])], 'd': 8}},
     'expected': sum(x for x in range(1,9))}
]
@pytest.mark.parametrize("contents, expected", [(test['contents'], test['expected']) for test in sum_tests])
def test_arraytainer_sum_method(contents, expected):
    contents = helpers.deepcopy_contents(contents)

    arraytainer = Arraytainer(contents)
    result = arraytainer.sum()

    helpers.assert_equal(result, expected)

@pytest.mark.parametrize("contents, expected", [(test['contents'], test['expected']) for test in sum_tests])
def test_jaxtainer_sum_method(contents, expected):
    contents = helpers.deepcopy_contents(contents, has_jax_arrays=True)

    jaxtainer = Jaxtainer(contents)
    result = jaxtainer.sum()

    helpers.assert_equal(result, jnp.array(expected))

sum_arrays_tests = [
    {'contents': {'a': np.array([1,2]), 'b': np.array([[3,4],[5,6]]), 'c': 7},
     'expected': np.array([[11, 13],[13, 15]]) },
    {'contents': [np.array([1,2]), np.array([[3,4],[5,6]]), 7],
     'expected': np.array([[11, 13],[13, 15]]) },
    {'contents': {'a': [np.array([1,2])], 'b': {'c': [np.array([[3,4],[5,6]])], 'd': 7}},
     'expected': np.array([[11, 13],[13, 15]]) }
]
@pytest.mark.parametrize("contents, expected", [(test['contents'], test['expected']) for test in sum_arrays_tests])
def test_arraytainer_sum_arrays_method(contents, expected):
    contents = helpers.deepcopy_contents(contents)

    arraytainer = Arraytainer(contents)
    result = arraytainer.sum_arrays()

    helpers.assert_equal(result, expected) 

@pytest.mark.parametrize("contents, expected", [(test['contents'], test['expected']) for test in sum_arrays_tests])
def test_jaxtainer_sum_arrays_method(contents, expected):
    contents = helpers.deepcopy_contents(contents, has_jax_arrays=True)

    jaxtainer = Jaxtainer(contents)
    result = jaxtainer.sum_arrays()

    helpers.assert_equal(result, jnp.array(expected)) 

sum_elems_tests = [
    {'contents': {'a': np.array([1,2]), 'b': np.array([[3,4],[5,6]]), 'c': 7},
     'expected': np.array([[11, 13],[13, 15]]) },
    {'contents': [np.array([1,2]), np.array([[3,4],[5,6]]), 7],
     'expected': np.array([[11, 13],[13, 15]]) },
    {'contents': {'a': [np.array([1,2])], 'b': np.array([[3,4],[5,6]]), 'c': 7},
     'expected': [np.array([[11, 13],[13, 15]])] }, 
    {'contents': {'a': np.array([1,2]), 'b': {'a':np.array([[3,4],[5,6]])}, 'c': 7},
     'expected': {'a': np.array([[11, 13],[13, 15]])} },
    {'contents': {'a': {'a':np.array([1,2]), 'b':np.array([[3,4],[5,6]])}, 'b': {'a': 1, 'b': 2}},
     'expected': {'a': np.array([2,3]), 'b':np.array([[5,6],[7,8]])} }, 
    {'contents': {'a': {'a':np.array([1,2]), 'b':np.array([[3,4],[5,6]])}, 'b': 2},
     'expected': {'a': np.array([3,4]), 'b':np.array([[5,6],[7,8]])} }, 
    {'contents': {'a': [{'a': np.array([[1,2],[3,4]])}, {'b': np.array([[5,6],[7,8]])}], 'b': [np.array([1,2]), np.array([2,3])]},
     'expected': [{'a': np.array([[2,4],[4,6]])}, {'b': np.array([[7,9],[9,11]])}] }
]
@pytest.mark.parametrize("contents, expected", [(test['contents'], test['expected']) for test in sum_elems_tests])
def test_arraytainer_sum_elems_method(contents, expected):
    contents = helpers.deepcopy_contents(contents)
    expected = helpers.deepcopy_contents(expected)

    arraytainer = Arraytainer(contents)
    result = arraytainer.sum_elems()
    if not isinstance(expected, np.ndarray):
        expected = Arraytainer(expected)

    helpers.assert_equal(result, expected) 

@pytest.mark.parametrize("contents, expected", [(test['contents'], test['expected']) for test in sum_elems_tests])
def test_jaxtainer_sum_elems_method(contents, expected):
    contents = helpers.deepcopy_contents(contents, has_jax_arrays=True)
    expected = helpers.deepcopy_contents(expected, has_jax_arrays=True)

    jaxtainer = Jaxtainer(contents)
    result = jaxtainer.sum_elems()
    if not isinstance(expected, jnp.ndarray):
        expected = Jaxtainer(expected)

    helpers.assert_equal(result, expected) 