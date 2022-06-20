import pytest
from arraytainers import Arraytainer, Jaxtainer
import numpy as np
import jax.numpy as jnp

import helpers

single_arg_input = {'a': {'b': [np.array([[1,2],[3,4]]), np.array([[1]])]}, 'c': [np.array([[2, 1],[6,7]]), np.array([[0.5,0],[0,0.5]])]}
single_arg_tests = [
 {'func': np.exp,  
  'expected': {'a': {'b': [np.exp([[1,2],[3,4]]), np.exp([[1]])]}, 'c': [np.exp([[2, 1],[6,7]]), np.exp([[0.5,0],[0,0.5]])]} },
 {'func': np.sin,  
  'expected': {'a': {'b': [np.sin([[1,2],[3,4]]), np.sin([[1]])]}, 'c': [np.sin([[2, 1],[6,7]]), np.sin([[0.5,0],[0,0.5]])]} },
 {'func': np.linalg.det,  
  'expected': {'a': {'b': [np.array(np.linalg.det([[1,2],[3,4]])), np.array(np.linalg.det([[1]]))]}, 'c': [np.array(np.linalg.det([[2, 1],[6,7]])), np.array(np.linalg.det([[0.5,0],[0,0.5]]))]} },
 {'func': np.linalg.inv,  
  'expected': {'a': {'b': [np.linalg.inv([[1,2],[3,4]]), np.linalg.inv([[1]])]}, 'c': [np.linalg.inv([[2, 1],[6,7]]), np.linalg.inv([[0.5,0],[0,0.5]])]} },
 {'func': np.shape,  
  'expected': {'a': {'b': [np.array([2,2]), np.array([1,1])]}, 'c': [np.array([2,2]), np.array([2,2])]} },
 {'func': lambda x: np.mean(x, axis=1),  
  'expected': {'a': {'b': [np.mean([[1,2],[3,4]], axis=1), np.mean([[1]], axis=1)]}, 'c': [np.mean([[2, 1],[6,7]], axis=1), np.mean([[0.5,0],[0,0.5]], axis=1)]} },
 {'func': lambda x: np.swapaxes(x, 1, 0),  
  'expected': {'a': {'b': [np.swapaxes([[1,2],[3,4]], 1, 0), np.swapaxes([[1]], 1, 0)]}, 'c': [np.swapaxes([[2, 1],[6,7]], 1, 0), np.swapaxes([[0.5,0],[0,0.5]], 1, 0)]} }
]
@pytest.mark.parametrize("func, expected", [(test['func'], test['expected']) for test in single_arg_tests])
def test_arraytainers_single_arg_numpy_funcs(func, expected):
    contents = helpers.deepcopy_contents(single_arg_input)
    expected = helpers.convert_arrays_to_numpy(expected)

    arraytainer = Arraytainer(contents)
    result = func(arraytainer)

    helpers.assert_equal(result.unpack(), expected, approx_equal=True)
    helpers.assert_equal(result, Arraytainer(expected), approx_equal=True)

@pytest.mark.parametrize("func, expected", [(test['func'], test['expected']) for test in single_arg_tests])
def test_jaxtainers_single_arg_funcs(func, expected):
    contents = helpers.deepcopy_contents(single_arg_input, has_jax_arrays=True)
    expected = helpers.deepcopy_contents(expected, has_jax_arrays=True)

    jaxtainer = Jaxtainer(contents)
    result = func(jaxtainer)

    helpers.assert_equal(result.unpack(), expected, approx_equal=True)
    helpers.assert_equal(result, Jaxtainer(expected), approx_equal=True)

binary_arg_input = {'a': {'b': [np.array([[1,2],[3,4]]), np.array(1)]}, 'c': [np.ones((2,2,2)), 0.5*np.ones((2,2))]}
scalar = 3.67
binary_op_scalar_tests = [
    {'func': lambda x,y: x+y, 
     'expected': {'a': {'b': [np.array([[1,2],[3,4]])+scalar, np.array(np.array(1)+scalar)]}, 'c': [np.ones((2,2,2))+scalar, 0.5*np.ones((2,2))+scalar]}},
    {'func': lambda x,y: x-y, 
     'expected': {'a': {'b': [np.array([[1,2],[3,4]])-scalar, np.array(np.array(1)-scalar)]}, 'c': [np.ones((2,2,2))-scalar, 0.5*np.ones((2,2))-scalar]}},
    {'func': lambda x,y: x*y, 
     'expected': {'a': {'b': [np.array([[1,2],[3,4]])*scalar, np.array(np.array(1)*scalar)]}, 'c': [np.ones((2,2,2))*scalar, 0.5*np.ones((2,2))*scalar]}},
    {'func': lambda x,y: x/y, 
     'expected': {'a': {'b': [np.array([[1,2],[3,4]])/scalar, np.array(np.array(1)/scalar)]}, 'c': [np.ones((2,2,2))/scalar, 0.5*np.ones((2,2))/scalar]}},
    {'func': lambda x,y: x**y, 
     'expected': {'a': {'b': [np.array([[1,2],[3,4]])**scalar, np.array((np.array(1))**scalar)]}, 'c': [np.ones((2,2,2))**scalar, (0.5*np.ones((2,2)))**scalar]}},
    {'func': lambda x,y: y+x, 
     'expected': {'a': {'b': [scalar+np.array([[1,2],[3,4]]), np.array(scalar+np.array(1))]}, 'c': [scalar+np.ones((2,2,2)), scalar+0.5*np.ones((2,2))]}},
    {'func': lambda x,y: y-x, 
     'expected': {'a': {'b': [scalar-np.array([[1,2],[3,4]]), np.array(scalar-np.array(1))]}, 'c': [scalar-np.ones((2,2,2)), scalar-0.5*np.ones((2,2))]}},
    {'func': lambda x,y: y*x, 
     'expected': {'a': {'b': [scalar*np.array([[1,2],[3,4]]), np.array(scalar*np.array(1))]}, 'c': [scalar*np.ones((2,2,2)), scalar*0.5*np.ones((2,2))]}},
    {'func': lambda x,y: y/x, 
     'expected': {'a': {'b': [scalar/np.array([[1,2],[3,4]]), np.array(scalar/(np.array(1)))]}, 'c': [scalar/np.ones((2,2,2)), scalar/(0.5*np.ones((2,2)))]}},
    {'func': lambda x,y: y**x, 
     'expected': {'a': {'b': [scalar**np.array([[1,2],[3,4]]), np.array(scalar**(np.array(1)))]}, 'c': [scalar**np.ones((2,2,2)), scalar**(0.5*np.ones((2,2)))]}},
]
@pytest.mark.parametrize("func, expected", [(test['func'], test['expected']) for test in binary_op_scalar_tests])
def test_arraytainers_binary_operators_with_scalar(func, expected):
    contents = helpers.deepcopy_contents(binary_arg_input)
    expected = helpers.deepcopy_contents(expected)

    arraytainer = Arraytainer(contents)
    result = func(arraytainer, scalar)

    helpers.assert_equal(result.unpack(), expected, approx_equal=True)
    helpers.assert_equal(result, Arraytainer(expected), approx_equal=True)

@pytest.mark.parametrize("func, expected", [(test['func'], test['expected']) for test in binary_op_scalar_tests])
def test_jaxtainers_binary_operators_with_scalar(func, expected):
    contents = helpers.deepcopy_contents(binary_arg_input, has_jax_arrays=True)
    expected = helpers.deepcopy_contents(expected, has_jax_arrays=True)

    jaxtainer = Jaxtainer(contents)
    result = func(jaxtainer, scalar)

    helpers.assert_equal(result.unpack(), expected, approx_equal=True)
    helpers.assert_equal(result, Jaxtainer(expected), approx_equal=True)

array = np.array([1.5, np.pi])
binary_op_array_tests = [
    {'func': lambda x,y: x+y, 
     'expected': {'a': {'b': [np.array([[1,2],[3,4]])+array, np.array(np.array(1)+array)]}, 'c': [np.ones((2,2,2))+array, 0.5*np.ones((2,2))+array]}},
    {'func': lambda x,y: x-y, 
     'expected': {'a': {'b': [np.array([[1,2],[3,4]])-array, np.array(np.array(1)-array)]}, 'c': [np.ones((2,2,2))-array, 0.5*np.ones((2,2))-array]}},
    {'func': lambda x,y: x*y, 
     'expected': {'a': {'b': [np.array([[1,2],[3,4]])*array, np.array(np.array(1)*array)]}, 'c': [np.ones((2,2,2))*array, 0.5*np.ones((2,2))*array]}},
    {'func': lambda x,y: x/y, 
     'expected': {'a': {'b': [np.array([[1,2],[3,4]])/array, np.array(np.array(1)/array)]}, 'c': [np.ones((2,2,2))/array, 0.5*np.ones((2,2))/array]}},
    {'func': lambda x,y: x**y, 
     'expected': {'a': {'b': [np.array([[1,2],[3,4]])**array, np.array((np.array(1))**array)]}, 'c': [np.ones((2,2,2))**array, (0.5*np.ones((2,2)))**array]}},
    {'func': lambda x,y: y+x, 
     'expected': {'a': {'b': [array+np.array([[1,2],[3,4]]), np.array(array+np.array(1))]}, 'c': [array+np.ones((2,2,2)), array+0.5*np.ones((2,2))]}},
    {'func': lambda x,y: y-x, 
     'expected': {'a': {'b': [array-np.array([[1,2],[3,4]]), np.array(array-np.array(1))]}, 'c': [array-np.ones((2,2,2)), array-0.5*np.ones((2,2))]}},
    {'func': lambda x,y: y*x, 
     'expected': {'a': {'b': [array*np.array([[1,2],[3,4]]), np.array(array*np.array(1))]}, 'c': [array*np.ones((2,2,2)), array*0.5*np.ones((2,2))]}},
    {'func': lambda x,y: y/x, 
     'expected': {'a': {'b': [array/np.array([[1,2],[3,4]]), np.array(array/(np.array(1)))]}, 'c': [array/np.ones((2,2,2)), array/(0.5*np.ones((2,2)))]}},
    {'func': lambda x,y: y**x, 
     'expected': {'a': {'b': [array**np.array([[1,2],[3,4]]), np.array(array**(np.array(1)))]}, 'c': [array**np.ones((2,2,2)), array**(0.5*np.ones((2,2)))]}},
]
@pytest.mark.parametrize("func, expected", [(test['func'], test['expected']) for test in binary_op_array_tests])
def test_arraytainer_binary_operators_with_array(func, expected):
    contents = helpers.deepcopy_contents(binary_arg_input)
    expected = helpers.deepcopy_contents(expected)

    arraytainer = Arraytainer(contents)
    result = func(arraytainer, array)

    helpers.assert_equal(result.unpack(), expected, approx_equal=True)
    helpers.assert_equal(result, Arraytainer(expected), approx_equal=True)

@pytest.mark.parametrize("func, expected", [(test['func'], test['expected']) for test in binary_op_array_tests])
def test_jaxtainer_binary_operators_with_array(func, expected):
    contents = helpers.deepcopy_contents(binary_arg_input, has_jax_arrays=True)
    expected = helpers.deepcopy_contents(expected, has_jax_arrays=True)

    jaxtainer = Jaxtainer(contents)
    jax_array = jnp.array(array)
    result = func(jaxtainer, jax_array)

    helpers.assert_equal(result.unpack(), expected, approx_equal=True)
    helpers.assert_equal(result, Jaxtainer(expected), approx_equal=True)

val_1 = -np.pi
val_2 = np.array([0.5, 2.33])
arraytainer_contents = {'a': val_1, 'c': val_2}
binary_op_arraytainer_tests = [
    {'func': lambda x,y: x+y, 
     'expected': {'a': {'b': [np.array([[1,2],[3,4]])+val_1, np.array(np.array(1)+val_1)]}, 'c': [np.ones((2,2,2))+val_2, 0.5*np.ones((2,2))+val_2]}},
    {'func': lambda x,y: x-y, 
     'expected': {'a': {'b': [np.array([[1,2],[3,4]])-val_1, np.array(np.array(1)-val_1)]}, 'c': [np.ones((2,2,2))-val_2, 0.5*np.ones((2,2))-val_2]}},
    {'func': lambda x,y: x*y, 
     'expected': {'a': {'b': [np.array([[1,2],[3,4]])*val_1, np.array(np.array(1)*val_1)]}, 'c': [np.ones((2,2,2))*val_2, 0.5*np.ones((2,2))*val_2]}},
    {'func': lambda x,y: x/y,
     'expected': {'a': {'b': [np.array([[1,2],[3,4]])/val_1, np.array(np.array(1)/val_1)]}, 'c': [np.ones((2,2,2))/val_2, 0.5*np.ones((2,2))/val_2]}},
    {'func': lambda x,y: x**y, 
     'expected': {'a': {'b': [np.array([[1,2],[3,4]])**val_1, np.array((np.array(1))**val_1)]}, 'c': [np.ones((2,2,2))**val_2, (0.5*np.ones((2,2)))**val_2]}},
    {'func': lambda x,y: y+x, 
     'expected': {'a': {'b': [val_1+np.array([[1,2],[3,4]]), np.array(val_1+np.array(1))]}, 'c': [val_2+np.ones((2,2,2)), val_2+0.5*np.ones((2,2))]}},
    {'func': lambda x,y: y-x, 
     'expected': {'a': {'b': [val_1-np.array([[1,2],[3,4]]), np.array(val_1-np.array(1))]}, 'c': [val_2-np.ones((2,2,2)), val_2-0.5*np.ones((2,2))]}},
    {'func': lambda x,y: y*x, 
     'expected': {'a': {'b': [val_1*np.array([[1,2],[3,4]]), np.array(val_1*np.array(1))]}, 'c': [val_2*np.ones((2,2,2)), val_2*0.5*np.ones((2,2))]}},
    {'func': lambda x,y: y/x, 
     'expected': {'a': {'b': [val_1/np.array([[1,2],[3,4]]), np.array(val_1/(np.array(1)))]}, 'c': [val_2/np.ones((2,2,2)), val_2/(0.5*np.ones((2,2)))]}},
    {'func': lambda x,y: y**x, 
     'expected': {'a': {'b': [val_1**np.array([[1,2],[3,4]]), np.array(val_1**(np.array(1)))]}, 'c': [val_2**np.ones((2,2,2)), val_2**(0.5*np.ones((2,2)))]}},
]

@pytest.mark.parametrize("func, expected", [(test['func'], test['expected']) for test in binary_op_arraytainer_tests])
def test_arraytainer_binary_operators_with_arraytainer(func, expected):
    contents = helpers.deepcopy_contents(binary_arg_input)
    other_contents = helpers.deepcopy_contents(arraytainer_contents)
    expected = helpers.deepcopy_contents(expected)

    arraytainer = Arraytainer(contents)
    other_arraytainer = Arraytainer(other_contents)
    result = func(arraytainer, other_arraytainer)

    helpers.assert_equal(result.unpack(), expected, approx_equal=True)
    helpers.assert_equal(result, Arraytainer(expected), approx_equal=True)

@pytest.mark.parametrize("func, expected", [(test['func'], test['expected']) for test in binary_op_arraytainer_tests])
def test_jaxtainer_binary_operators_with_jaxtainer(func, expected):
    contents = helpers.deepcopy_contents(binary_arg_input, has_jax_arrays=True)
    other_contents = helpers.deepcopy_contents(arraytainer_contents, has_jax_arrays=True)
    expected = helpers.deepcopy_contents(expected, has_jax_arrays=True)

    jaxtainer = Jaxtainer(contents)
    other_jaxtainer = Jaxtainer(other_contents)
    result = func(jaxtainer, other_jaxtainer)

    helpers.assert_equal(result.unpack(), expected, approx_equal=True)
    helpers.assert_equal(result, Jaxtainer(expected), approx_equal=True)

einsum_tests = [
    {'arg1': {'a': [np.ones((3,3,3)), np.ones((2,2,2))], 'b': np.ones((3,2,3))}, 
     'arg2': {'a': [np.ones((3,3,3)), np.ones((2,2,2))], 'b': np.ones((3,2,3))}, 
     'einstr': 'ijk,klm->ijlm', 
     'expected':  {'a': [3*np.ones((3,3,3,3)), 2*np.ones((2,2,2,2))], 'b': 3*np.ones((3,2,2,3))}},
    {'arg1': {'a': [np.ones((3,3,3)), np.ones((2,2,2))], 'b': np.ones((3,2))}, 
     'arg2': {'a': [np.ones((3,3,3)), np.ones((2,2,2))], 'b': np.ones((2,3))}, 
     'einstr': {'a': 'ijk,klm->ijlm', 'b': 'ij,jk->ik'}, 
     'expected':  {'a': [3*np.ones((3,3,3,3)), 2*np.ones((2,2,2,2))], 'b': 2*np.ones((3,3))}},
]
@pytest.mark.parametrize("arg1, arg2, einstr, expected", [(test['arg1'], test['arg2'], test['einstr'], test['expected']) for test in einsum_tests])
def test_arraytainer_einsum(arg1, arg2, einstr, expected):
    arg1 = helpers.deepcopy_contents(arg1)
    arg2 = helpers.deepcopy_contents(arg2)
    expected = helpers.deepcopy_contents(expected)
    arraytainer_1 = Arraytainer(arg1)
    arraytainer_2 = Arraytainer(arg2)
    if not isinstance(einstr, str):
        einstr = Arraytainer(einstr, array_conversions=False)

    result = np.einsum(einstr, arraytainer_1, arraytainer_2)

    helpers.assert_equal(result.unpack(), expected, approx_equal=True)
    helpers.assert_equal(result, Arraytainer(expected), approx_equal=True)

@pytest.mark.parametrize("arg1, arg2, einstr, expected", [(test['arg1'], test['arg2'], test['einstr'], test['expected']) for test in einsum_tests])
def test_jaxtainer_einsum(arg1, arg2, einstr, expected):
    arg1 = helpers.deepcopy_contents(arg1, has_jax_arrays=True)
    arg2 = helpers.deepcopy_contents(arg2, has_jax_arrays=True)
    expected = helpers.deepcopy_contents(expected, has_jax_arrays=True)
    arraytainer_1 = Jaxtainer(arg1)
    arraytainer_2 = Jaxtainer(arg2)
    if not isinstance(einstr, str):
        einstr = Jaxtainer(einstr, array_conversions=False)
        
    result = np.einsum(einstr, arraytainer_1, arraytainer_2)

    helpers.assert_equal(result.unpack(), expected, approx_equal=True)
    helpers.assert_equal(result, Jaxtainer(expected), approx_equal=True)

concatenate_tests = [
    {'arg1': {'a': [np.ones((3,3,3)), np.ones((2,2,2))], 'b': np.ones((3,2,3))}, 
     'arg2': {'a': [np.ones((3,3,3)), np.ones((2,2,2))], 'b': np.ones((3,2,3))}, 
     'axis': 1, 
     'expected':  {'a': [np.ones((3,6,3)), np.ones((2,4,2))], 'b': np.ones((3,4,3))}},
    {'arg1': {'a': [np.ones((3,3,3)), np.ones((2,2,2))], 'b': np.ones((3,2))}, 
     'arg2': {'a': [np.ones((3,3,3)), np.ones((2,2,2))], 'b': np.ones((5,2))}, 
     'axis': {'a': 2, 'b': 0}, 
     'expected':  {'a': [np.ones((3,3,6)), np.ones((2,2,4))], 'b': np.ones((8,2))}},
]
@pytest.mark.parametrize("arg1, arg2, axis, expected", [(test['arg1'], test['arg2'], test['axis'], test['expected']) for test in concatenate_tests])
def test_arraytainer_concatenate(arg1, arg2, axis, expected):
    arg1 = helpers.deepcopy_contents(arg1)
    arg2 = helpers.deepcopy_contents(arg2)
    expected = helpers.deepcopy_contents(expected)
    arraytainer_1 = Arraytainer(arg1)
    arraytainer_2 = Arraytainer(arg2)
    if not isinstance(axis, int):
        axis = Arraytainer(axis, array_conversions=False)
        
    result = np.concatenate((arraytainer_1, arraytainer_2), axis=axis)

    helpers.assert_equal(result.unpack(), expected, approx_equal=True)
    helpers.assert_equal(result, Arraytainer(expected), approx_equal=True)

@pytest.mark.parametrize("arg1, arg2, axis, expected", [(test['arg1'], test['arg2'], test['axis'], test['expected']) for test in concatenate_tests])
def test_jaxtainer_concatenate(arg1, arg2, axis, expected):
    arg1 = helpers.deepcopy_contents(arg1, has_jax_arrays=True)
    arg2 = helpers.deepcopy_contents(arg2, has_jax_arrays=True)
    expected = helpers.deepcopy_contents(expected, has_jax_arrays=True)
    jaxtainer_1 = Jaxtainer(arg1)
    jaxtainer_2 = Jaxtainer(arg2)
    if not isinstance(axis, int):
        axis = Jaxtainer(axis, array_conversions=False)
        
    result = np.concatenate((jaxtainer_1, jaxtainer_2), axis=axis)

    helpers.assert_equal(result.unpack(), expected, approx_equal=True)
    helpers.assert_equal(result, Jaxtainer(expected), approx_equal=True)

stack_tests = [
    {'arg': {'a': [np.ones((3,3)), np.ones((2,2,2))], 'b': np.ones((1,))}, 
     'num': 5,
     'axis': 1, 
     'expected':  {'a': [np.ones((3,5,3)), np.ones((2,5,2,2))], 'b': np.ones((1,5))}},
    {'arg': {'a': [np.ones((3,3)), np.ones((2,2,2))], 'b': np.ones((1,))},  
     'num': 4,
     'axis': {'a': 2, 'b': 0}, 
     'expected': {'a': [np.ones((3,3,4)), np.ones((2,2,4,2))], 'b': np.ones((4,1))}},
]
@pytest.mark.parametrize("arg, num, axis, expected", [(test['arg'], test['num'], test['axis'], test['expected']) for test in stack_tests])
def test_arraytainer_stack(arg, num, axis, expected):
    arg = helpers.deepcopy_contents(arg)
    expected = helpers.deepcopy_contents(expected)
    arraytainer = Arraytainer(arg)
    if not isinstance(axis, int):
        axis = Arraytainer(axis, array_conversions=False)

    result = np.stack([arraytainer for _ in range(num)], axis=axis)

    helpers.assert_equal(result.unpack(), expected, approx_equal=True)
    helpers.assert_equal(result, Arraytainer(expected), approx_equal=True)

@pytest.mark.parametrize("arg, num, axis, expected", [(test['arg'], test['num'], test['axis'], test['expected']) for test in stack_tests])
def test_jaxtainer_stack(arg, num, axis, expected):
    arg = helpers.deepcopy_contents(arg, has_jax_arrays=True)
    expected = helpers.deepcopy_contents(expected, has_jax_arrays=True)
    jaxtainer = Jaxtainer(arg)
    if not isinstance(axis, int):
        axis = Jaxtainer(axis, array_conversions=False)

    result = np.stack([jaxtainer for _ in range(num)], axis=axis)

    helpers.assert_equal(result.unpack(), expected, approx_equal=True)
    helpers.assert_equal(result, Jaxtainer(expected), approx_equal=True)

split_tests = [
    {'arg': {'a': [np.ones((2,6)), np.ones((2,3,2))], 'b': np.ones((1,9))}, 
     'sections': 3,
     'axis': 1, 
     'expected':  {'a': [3*[np.ones((2,2))], 3*[np.ones((2,1,2))]], 'b': 3*[np.ones((1,3))]}},
    {'arg': {'a': [np.ones((2,4)), np.ones((2,6,2))], 'b': [np.ones((1,9)), np.ones((3,3,3))]},  
     'sections': {'a': 2, 'b': 3},
     'axis': 1, 
     'expected': {'a': [2*[np.ones((2,2))], 2*[np.ones((2,3,2))]], 'b': [3*[np.ones((1,3))], 3*[np.ones((3,1,3))]]}},
    {'arg': {'a': [np.ones((2,4)), np.ones((2,6,2))], 'b': [np.ones((10,1)), np.ones((4,3,3))]},  
     'sections': 2,
     'axis': {'a': 1, 'b': 0}, 
     'expected': {'a': [2*[np.ones((2,2))], 2*[np.ones((2,3,2))]], 'b': [2*[np.ones((5,1))], 2*[np.ones((2,3,3))]]}},
    {'arg': {'a': [np.ones((2,4)), np.ones((2,6,2))], 'b': [np.ones((9,1)), np.ones((3,3,3))]},  
     'sections': {'a': 2, 'b': 3},
     'axis': {'a': 1, 'b': 0}, 
     'expected': {'a': [2*[np.ones((2,2))], 2*[np.ones((2,3,2))]], 'b': [3*[np.ones((3,1))], 3*[np.ones((1,3,3))]]}}
]
@pytest.mark.parametrize("arg, sections, axis, expected", [(test['arg'], test['sections'], test['axis'], test['expected']) for test in split_tests])
def test_arraytainer_split(arg, sections, axis, expected):
    arg = helpers.deepcopy_contents(arg)
    expected = helpers.deepcopy_contents(expected)
    arraytainer = Arraytainer(arg)
    if not isinstance(sections, int):
        sections = Arraytainer(sections, array_conversions=False)
    if not isinstance(axis, int):
        axis = Arraytainer(axis, array_conversions=False)

    result = np.split(arraytainer, sections, axis=axis)
    
    helpers.assert_equal(result.unpack(), expected, approx_equal=True)
    helpers.assert_equal(result, Arraytainer(expected), approx_equal=True)

@pytest.mark.parametrize("arg, sections, axis, expected", [(test['arg'], test['sections'], test['axis'], test['expected']) for test in split_tests])
def test_jaxtainer_split(arg, sections, axis, expected):
    arg = helpers.deepcopy_contents(arg, has_jax_arrays=True)
    expected = helpers.deepcopy_contents(expected, has_jax_arrays=True)
    jaxtainer = Jaxtainer(arg)
    if not isinstance(sections, int):
        sections = Jaxtainer(sections, array_conversions=False)
    if not isinstance(axis, int):
        axis = Jaxtainer(axis, array_conversions=False)

    result = np.split(jaxtainer, sections, axis=axis)
    
    helpers.assert_equal(result.unpack(), expected, approx_equal=True)
    helpers.assert_equal(result, Jaxtainer(expected), approx_equal=True)