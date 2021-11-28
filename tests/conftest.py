import pytest
from itertools import product
import numpy as np

STD_SHAPES = \
{'empty': [{}, []],
 'single_arrays': [[(2,2)], {'a': (2,2)}],
 'single_elements': [[{'a':(1,2), 'b':[(2,3)]}], {'a':[{'b':[(3,2)]}]}],
 'simple_dicts': [{'a':(1,),'b':(2,),'c':(3,)}, {'a':(2,2),'b':(3,2)}],
 'simple_lists': [[(1,),(2,),(3,)], [(1,2),(3,2)]],
 'dicts_with_strange_keys': [{'a':(3,), 1:(2,), 1.5:(2,), ('a',1): (1,)}, {'a':(1,), 1:(2,), 1.5:(3,), ('a', 1): (4,)}],
 'nested_lists': [[[(2,1)], [(2,1)]], [[(2,3,2), (3,), [(2,), (3,2,1)]], (1,)]],
 'nested_dicts': [{'a':{'a':(2,3), 'b':(3,), 'c':{'a':(2,), 'b':(3,)}},'b':(1,)}],
 'dict_of_lists': [{'a': [(1,),(2,)], 'b': [(1,),(2,),(3,)]}, {'a': [(1,2),(2,3)], 'b': [(2,3),(4,),(5,6)]}],
 'list_of_dicts': [[{'a':(1,2),'b':(2,3)}, {'b':(2,3),'c':(4,),'d':(5,6)}]],
 'mixed': [{'a':[{'a':(1,2),'b':(1,)}, {'a':(1,2)}], 'b':[{'c':(1,2)}]}, 
           [{'a':(2,2), 'b': (1,3)}, {'c':[(1,2),(2,3)]}, {'a':{'d':(2,2)}}]]}

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

@pytest.fixture(scope="function", params=[val_i for val in STD_SHAPES.values() for val_i in val], 
                                     ids=[key for key, val in STD_SHAPES.items() for _ in val])
def std_contents(request):
    yield create_contents(request.param)

BOOL_VALS = {'true': 1, 'false': 0, 'true&false': None}
BOOL_SHAPES = {f'{key}_{bool_keys}': tuple(product(val, [bool_vals])) 
               for key, val in STD_SHAPES.items() for bool_keys, bool_vals in BOOL_VALS.items()}
               
@pytest.fixture(scope="function", params=[val_i for val in BOOL_SHAPES.values() for val_i in val], 
                                     ids=[key for key, val in BOOL_SHAPES.items() for _ in val])
def bool_contents(request):
    shapes, single_value = request.param
    if single_value is not None:
        contents = create_contents(shapes, single_value=single_value, return_bool=True)
    else:
        contents = create_contents(shapes, return_bool=True)
    yield contents