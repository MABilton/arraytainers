import pytest
import numpy as np
import jax
import jax.numpy as jnp
from arraytainers import Jaxtainer

import helpers

# Note: bug in Jax with tree_flatten() with dictionaries with strings and ints:
vmap_tests = [
    {'contents': {'a': jnp.array([3, 4]), '0': jnp.array([[1, 2],[3, 4]])},
     'function': lambda x: x['a']**2 + jnp.log(x['0']),
     'expected': np.array([[3], [4]])**2 + np.log([[1, 2],[3, 4]]) }, 
    {'contents': {'a': [jnp.array([3, 4])], '0': jnp.array([[1, 2],[3, 4]])},
     'function': lambda x: x['a']**2 + jnp.log(x['0']),
     'expected': [np.array([[3], [4]])**2 + np.log([[1, 2],[3, 4]])] },
    {'contents': {'a': {'b': jnp.array([3, 4])}, '0': {'b': [jnp.array([[3, 4],[5, 6]])]}},
     'function': lambda x: x['a']**2 + np.log(x['0']),
     'expected': {'b': [np.array([[3], [4]])**2 + np.log([[3, 4],[5, 6]])]} }
]
@pytest.mark.parametrize("contents, function, expected", [(test['contents'], test['function'], test['expected']) for test in vmap_tests])
def test_jaxtainer_input_to_vmap_func(contents, function, expected):
    contents = helpers.deepcopy_contents(contents, has_jax_arrays=True)
    vmapped_func = jax.vmap(function, in_axes=0)
    expected = helpers.deepcopy_contents(expected, has_jax_arrays=True)

    if isinstance(expected, (list,dict)):
        expected = Jaxtainer(expected) 

    jaxtainer = Jaxtainer(contents)
    output = vmapped_func(jaxtainer)

    helpers.assert_equal(output, expected, approx_equal=True)
    
jacfwd_tests = [
    {'contents': {'a': jnp.array([1, 2]), '0': jnp.array([2, 3])},
     'function': lambda x: jnp.sum(x['a']**2 + jnp.log(x['0'])),
     'expected': {'a': jnp.array([2, 4]), '0': jnp.array([1/2, 1/3]) } }, 
    {'contents': {'a': [{'c': jnp.array([1, 2])}], '0': jnp.array([2, 3])},
     'function': lambda x: jnp.sum(x['a'][0]['c']**2 + jnp.log(x['0'])),
     'expected': {'a': [{'c': jnp.array([2, 4])}], '0': jnp.array([1/2, 1/3]) } }, 
    {'contents': {'a': [{'c': jnp.array([1, 2])}], '0': jnp.array([2, 3])},
     'function': lambda x: x['a'][0]['c']**2 + jnp.log(x['0']),
     'expected': {'a': [{'c': jnp.array([[2,0],[0,4]])}], '0': jnp.array([[1/2,0],[0,1/3]]) } }, 
    {'contents': {'a': [{'c': jnp.array([1, 2])}], '0': jnp.array([2, 3])},
     'function': lambda x: x['a']**2 + jnp.log(x['0']), 
     'expected': [{'c': {'a': [{'c': jnp.array([[2,0],[0,4]])}], '0': jnp.array([[1/2,0],[0,1/3]])}}] }
]
@pytest.mark.parametrize("contents, function, expected", [(test['contents'], test['function'], test['expected']) for test in jacfwd_tests])
def test_jaxtainer_input_to_jacfwd_func(contents, function, expected):
    contents = helpers.deepcopy_contents(contents, has_jax_arrays=True, dtype=float)
    jacfwd_func = jax.jacfwd(function)
    expected = helpers.deepcopy_contents(expected, has_jax_arrays=True, dtype=float)

    jaxtainer = Jaxtainer(contents)
    output = jacfwd_func(jaxtainer)

    helpers.assert_equal(output, Jaxtainer(expected), approx_equal=True)

jacfwd_and_vmap_tests = [
    # Func returns scalar:
    {'contents': {'a': jnp.array([[1,2],[3,4]]), '0': jnp.array([2, 3])},
     'function': lambda x: jnp.sum(x['a']**2 + jnp.log(x['0'])),
     # Need 2* for broadcasting:
     'expected': {'a': jnp.array([[2,4],[6,8]]), '0': 2*jnp.array([1/2, 1/3]) } }, 
    {'contents': [jnp.array([[1,2],[3,4]]), jnp.array([2, 3])],
     'function': lambda x: jnp.sum(x[0]**2 + jnp.log(x[1])),
     'expected': [jnp.array([[2,4],[6,8]]), 2*jnp.array([1/2, 1/3])] }, 
    {'contents': {'a': [{'c': jnp.array([[1,2],[3,4]])}], '0': jnp.array([2, 3])},
     'function': lambda x: jnp.sum(x['a'][0]['c']**2 + jnp.log(x['0'])),
     'expected': {'a': [{'c': jnp.array([[2,4],[6,8]])}], '0': 2*jnp.array([1/2, 1/3]) } }, 
    
    # Func returns array:
    {'contents': {'a': jnp.array([[1,2],[3,4]]), '0': jnp.array([2,3])},
     'function': lambda x: x['a']**2 + jnp.log(x['0']),
     'expected': {'a': jnp.array([[[2.,0.],[0.,4.]],[[6.,0.],[0.,8.]]]), '0': jnp.array([[1/2,1/2],[1/3,1/3]])} }, 
    {'contents': [jnp.array([[1,2],[3,4]]), jnp.array([2,3])],
     'function': lambda x: x[0]**2 + jnp.log(x[1]),
     'expected': [jnp.array([[[2,0],[0,4]],[[6,0],[0,8]]]), jnp.array([[1/2,1/2],[1/3,1/3]])] }, 
    {'contents': {'a': [{'c': jnp.array([[1,2],[3,4]])}], '0': jnp.array([2, 3])},
     'function': lambda x: x['a'][0]['c']**2 + jnp.log(x['0']),
     'expected': {'a': [{'c': jnp.array([[[2,0],[0,4]],[[6,0],[0,8]]])}], '0': jnp.array([[1/2,1/2],[1/3,1/3]])} }, 
    
    # Func returns arraytainer:
    {'contents': {'a': [{'c': jnp.array([[1,2],[3,4]])}], '0': jnp.array([2, 3])},
     'function': lambda x: x['a']**2 + jnp.log(x['0']),
     'expected': [{'c': {'a': [{'c': jnp.array([[[2,0],[0,4]],[[6,0],[0,8]]])}], '0': jnp.array([[1/2,1/2],[1/3,1/3]])} }] }, 
    {'contents': {'a': [{'c': jnp.array([[1,2],[3,4]])}], '0': [jnp.array([2, 3])]},
     'function': lambda x: x['a']**2 + np.log(x['0']),
     'expected': [{'c': {'a': [{'c': jnp.array([[[2,0],[0,4]],[[6,0],[0,8]]])}], '0': [jnp.array([[1/2,1/2],[1/3,1/3]])]}}] }
]
@pytest.mark.parametrize("contents, function, expected", [(test['contents'], test['function'], test['expected']) for test in jacfwd_and_vmap_tests])
def test_jaxtainer_input_to_jacfwd_and_vmap_func(contents, function, expected):
    contents = helpers.deepcopy_contents(contents, has_jax_arrays=True, dtype=float)
    jacfwd_func = jax.vmap(jax.jacfwd(function), in_axes=0)
    expected = helpers.deepcopy_contents(expected, has_jax_arrays=True, dtype=float)

    jaxtainer = Jaxtainer(contents)
    output = jacfwd_func(jaxtainer)

    helpers.assert_equal(output, Jaxtainer(expected), approx_equal=True)

jacfwd_and_vmap_constructor_tests = [
    {'contents': jnp.array([[1,2,3],[2,3,4]]),
     'function': lambda x: Jaxtainer({'a': x[0], 'b': x[1], 'c': x[2]})**2,
     'expected': {'a': jnp.array([[2,0,0],[4,0,0]]), 'b': jnp.array([[0,4,0],[0,6,0]]), 'c': jnp.array([[0,0,6],[0,0,8]])} },
    {'contents': jnp.array([[1,2,3],[2,3,4]]),
     'function': lambda x: Jaxtainer([x[0], x[1], x[2]])**2,
     'expected': [jnp.array([[2,0,0],[4,0,0]]), jnp.array([[0,4,0],[0,6,0]]), jnp.array([[0,0,6],[0,0,8]])] },
    {'contents': jnp.array([[1,2,3],[2,3,4]]),
     'function': lambda x: Jaxtainer({'a': [x[0], {'c':x[1]}], 'b': {'a': x[2]}})**2,
     'expected': {'a': [jnp.array([[2,0,0],[4,0,0]]), {'c': jnp.array([[0,4,0],[0,6,0]])}], 'b': {'a': jnp.array([[0,0,6],[0,0,8]])}} }
]
@pytest.mark.parametrize("contents, function, expected", [(test['contents'], test['function'], test['expected']) for test in jacfwd_and_vmap_constructor_tests])
def test_create_jaxtainer_in_jacfwd_and_vmap_func(contents, function, expected):
    contents = helpers.deepcopy_contents(contents, has_jax_arrays=True, dtype=float)
    expected = helpers.deepcopy_contents(expected, has_jax_arrays=True, dtype=float)

    output = jax.vmap(jax.jacfwd(function))(contents)

    helpers.assert_equal(output, Jaxtainer(expected), approx_equal=True)

from_array_tests = [
    {'array': jnp.array([[1,2],[2,3]]),
     'shape': {'a': 1, 'b': 1},
     'function': lambda x, shape: Jaxtainer.from_array(x, shape)**2,
     'expected': {'a': jnp.array([ [[2,0]], [[4,0]] ]), 'b': jnp.array([ [[0,4]], [[0,6]] ])}},
    {'array': jnp.array([[1,2],[2,3]]),
     'shape': [1, 1],
     'function': lambda x, shape: Jaxtainer.from_array(x, shape)**2,
     'expected': [jnp.array([ [[2,0]], [[4,0]] ]), jnp.array([ [[0,4]], [[0,6]] ])] },
    {'array': jnp.array([[1,2],[2,3]]),
     'shape': {'a': [{'b': 1}], 'c': 1},
     'function': lambda x, shape: Jaxtainer.from_array(x, shape)**2,
     'expected': {'a': [{'b': jnp.array([ [[2,0]], [[4,0]] ])}], 'c': jnp.array([ [[0,4]], [[0,6]] ])} }
]
@pytest.mark.parametrize("array, shape, function, expected", [(test['array'], test['shape'], test['function'], test['expected']) for test in from_array_tests])
def test_from_array_in_jacfwd_and_vmap_func(array, shape, function, expected):
    shape = helpers.deepcopy_contents(shape, has_jax_arrays=True, dtype=float)
    expected = helpers.deepcopy_contents(expected, has_jax_arrays=True, dtype=float)
    array = jnp.array(array, dtype=float)

    shape = Jaxtainer(shape)
    output = jax.vmap(jax.jacfwd(function,argnums=0),in_axes=(0,None))(array, shape)

    helpers.assert_equal(output, Jaxtainer(expected), approx_equal=True)

method_tests = [
    {'contents': {'a': jnp.array([1]), 'b': jnp.array([2])},
     'function': lambda x: x.flatten()**2,
     'expected': {'a': jnp.array([[2,0]]), 'b': jnp.array([[0,4]])} },
    {'contents': {'a': jnp.array([ [[1]] ]), 'b': jnp.array([ [[2]] ])},
     'function': lambda x: x.squeeze()**2,
     'expected': {'a': {'a': jnp.array([ [[2]] ]), 'b': jnp.array([ [[0]] ])}, 'b': {'a': jnp.array([ [[0]] ]), 'b': jnp.array([ [[4]] ])}} },
    {'contents': {'a': jnp.array([1]), 'b': jnp.array([2])},
     'function': lambda x: (x**2).sum(),
     'expected': {'a': jnp.array([2]), 'b': jnp.array([4])} },
    {'contents': {'a': jnp.array([1]), 'b': jnp.array([2])},
     'function': lambda x: x.first_array**2,
     'expected': {'a': jnp.array([2]), 'b': jnp.array([0])} },
    {'contents': {'a': {'c': jnp.array([1])}, 'b': jnp.array([2])},
     'function': lambda x: (x**2).sum_elems(),
     'expected': {'c': {'a': {'c': jnp.array([2])}, 'b': jnp.array([4])} } },
    {'contents': {'a': jnp.array([[1,2]]), 'b': jnp.array([[2,3]])},
     'function': lambda x: (x**2).sum_arrays(),
     'expected': {'a': jnp.array([ [[2,0],[0,4]] ]), 'b': jnp.array([ [[4,0],[0,6]] ])} }
]
@pytest.mark.parametrize("contents, function, expected", [(test['contents'], test['function'], test['expected']) for test in method_tests])
def test_jaxtainer_method_in_jacfwd_and_vmap_func(contents, function, expected):
    contents = helpers.deepcopy_contents(contents, has_jax_arrays=True, dtype=float)
    expected = helpers.deepcopy_contents(expected, has_jax_arrays=True, dtype=float)

    jaxtainer = Jaxtainer(contents)
    output = jax.vmap(jax.jacfwd(function))(jaxtainer)

    helpers.assert_equal(output, Jaxtainer(expected), approx_equal=True)