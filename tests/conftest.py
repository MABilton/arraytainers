import pytest
import itertools
import numpy as np

STD_SHAPES = ({}, [], # Empty
              [(2,2)], {'a': (2,2)}, # Single values
              {'a':(1,),'b':(2,),'c':(3,)}, {'a':(2,2),'b':(3,2)}, # Simple dicts
              [(1,),(2,),(3,)], [(1,2),(3,2)], # Simple lists
              {'a':(3,), 1:(2,), 1.5:(2,), ('a',1): (1,)}, 
              {'a':(1,), 1:(2,), 1.5:(3,), ('a', 1): (4,)}, # Dicts with 'strange' keys
              [[(2,1)], [(2,1)]], [[(2,3,2), (3,), [(2,), (3,2,1)]], (1,)], # Nested lists
              {'a':{'a':(2,1)},'b':{'c':(2,1)}}, 
              {'a':{'a':(2,3), 'b':(3,), 'c':{'a':(2,), 'b':(3,)}},'b':(1,)}, # Nested dicts
              {'a': [(1,),(2,)], 'b': [(1,),(2,),(3,)]}, {'a': [(1,2),(2,3)], 'b': [(2,3),(4,),(5,6)]}, # Dict of lists
              [{'a':(1,2),'b':(2,3)}, {'b':(2,3),'c':(4,),'d':(5,6)}], # List of dicts
              {'a':[{'a':(1,2),'b':(1,)}, {'a':(1,2)}], 'b':[{'c':(1,2)}]}) # Dict of lists of dicts

def create_contents(shapes, single_value=None, return_bool=False, seed=42):
    np.random.seed(seed)
    contents = create_contents_recursion(shapes, single_value, return_bool)
    return contents

def create_contents_recursion(shapes, single_value, return_bool):

    if isinstance(shapes, dict):
        return {key: create_contents(val) for key, val in shapes.items()}

    elif isinstance(shapes, list):
        return [create_contents(val) for val in shapes]

    elif isinstance(shapes, tuple):
        if single_value is not None:
            array = single_value*np.ones(*shapes)
            array = array.astype(np.bool) if return_bool else array
        else:
            array = np.random.rand(*shapes)
            array = array>0.5 if return_bool else 10*array
        return array

@pytest.fixture(scope="function")
def contents(request):
    contents = create_contents(request.param)
    yield contents

@pytest.fixture(scope="function", params=STD_SHAPES)
def std_contents(request):
    yield create_contents(request.param)

@pytest.fixture(scope="function", params=itertools.product(STD_SHAPES, [0,1,None]))
def bool_contents(request):
    shapes, single_value = request.param
    if single_value is not None:
        contents = create_contents(shapes, single_value=single_value, return_bool=True)
    else:
        contents = create_contents(shapes, return_bool=True)
    yield contents

# @pytest.fixture(scope="function", params=SHAPES)
# def key_contents(request):
#     key_shapes = request.param
#     contents = create_contents(key_shapes, return_bool=True)
#     yield contents

# @pytest.fixture(scope="function", params=SHAPES)
# def val_contents(request):
#     key_shapes = request.param
#     contents = create_contents(key_shapes)
#     yield contents

# @pytest.fixture(scope="function", params=)
# def non_contents(request):
#     val = request.param
#     yield val

# @pytest.fixture(scope="function", params=)
# def array_or_slice(request):
#     array_or_slice = request.param
#     if not isinstance(array_or_slice, slice):
#         array_or_slice = np.array(array_or_slice)
#     yield array_or_slice

# FUNCS = ('einsum': np.einsum, 'solve': np.linalg.solve, 'matmult': lambda x, y: x @ y, 'kron': np.linalg.kron)

# KWARGS = ('einsum': 'ika,bik->ab', 'solve':None, 'matmult':None, 'kron':None)
# ARG_SHAPES = (,)

# def create_func_and_args_params():
#     # Create einsum shapes:
#     es_arrays = {'valid': {'arg_1':(3,2,1), 'arg_2':(4,3,2)}, 
#                  'invalid_sizes': {'arg_1':(2,2,1), 'arg_2':(4,3,2)}, 
#                  'invalid_dims': {'arg_1':(3,2,1), 'arg_2':(3,2)}}
#     es_correct_keys = \
#     itertools.chain.from_iterable([[tuple({'a':shape, 'b':shape} for shape in shapes),
#                                     tuple([shape, shape] for shape in shapes),
#                                     tuple({'a': {'a':shape, 'b':shape}, 'b':shape} for shape in shapes),
#                                     tuple([{'a':[shape, shape]}, {'b':[shape, shape]}, shape] for shape in shapes)] 
#                                     for shapes in es_arrays.values()])
#     shape_1, shape_2 = es_arrays['valid'].values()
#     es_incorrect_keys = [tuple([{'a':shape_1}, [shape_2]]),
#                          tuple([{'a':shape_1, 'b':shape_1}, [shape_2]]),
#                          tuple([[shape_1], [shape_2, shape_2]]),
#                          tuple([{'a':shape_1}, {'b':shape_1}])]
#     es_shapes = es_correct_keys + es_incorrect_keys

#     # Create shapes for ufunc operations:


#     # Create shapes for all other function calls:

#     return

# @pytest.fixture(scope="function", params=zip(ARG_SHAPES, FUNCS))
# def func_and_args(request):


#     func, args, kwargs = request.param
#     yield (func, args, kwargs)
