import pytest
from itertools import product
import numpy as np
import jax.numpy as jnp
from copy import deepcopy
from main_tests import utils
from main_tests.utils import create_contents

STD_SHAPES = {
'empty_arraytainers': [ {}, [] ],
'empty_arrays': [ {'a':(0,),'b':(0,)}, [(0,),(0,),{'c':(0,)}] ],
'single_arrays': [ [(2,2)], {'a': (2,2)} ],
'single_elements': [ [{'a':(1,2), 'b':[(2,3)]}], {'a':[{'b':[(3,2)]}]} ],
'simple_dicts': [ {'a':(1,),'b':(2,),'c':(3,)}, {'a':(2,2),'b':(3,2)}],
'simple_lists': [ [(1,),(2,2),(3,3,3)], [(1,2),(3,2)] ],
# 'dicts_with_strange_keys': [ {'a':(3,2), 1:(2,3), 1.5:(2,1), ('a',1): (1,)} ],
'nested_lists': [ [[(2,1)], [(2,1)]], [[(2,3,2), (3,), [(2,), (3,2,1)]], (1,)] ],
'nested_dicts': [ {'a':{'a':(2,3), 'b':(3,), 'c':{'a':(2,), 'b':(3,)}},'b':(1,)} ],
'dict_of_lists': [ {'a': [(1,3),(2,)], 'b': [(1,),(2,3,2,3),(3,3,2)]} ],
'list_of_dicts': [ [{'a':(1,2),'b':(2,3)}, {'b':(2,3),'c':(4,),'d':(5,6)}] ],
'mixed': [ {'a':[{'a':(3,1,2),'b':(1,)}, {'a':(3,1,2)}], 'b':[{'c':(1,2)}]}, 
           [{'a':(2,2), 'b': (1,3)}, {'c':[(1,2,2),(2,3)]}, {'a':{'d':(2,2)}}] ]
}

@pytest.fixture(scope="function", params=[])
def contents(request):
    array_constructor = request.cls.array_constructor
    contents_out = create_contents(request.param, array_constructor)
    yield contents_out

@pytest.fixture(scope="function")
def array(request):
    array_data = request.param
    array_constructor = request.cls.array_constructor
    # Interpret a tuple as an array shape:
    if isinstance(array_data, tuple):
        array = create_contents(array_data, array_constructor)
    # Interpret a list as a set of values from which to form an array:
    elif isinstance(array_data, list):
        array = array_constructor(array_data)
    yield array

@pytest.fixture(scope="function")
def contents_list(request):
    array_constructor = request.cls.array_constructor
    items = request.param
    contents_out = [create_contents(item_i, array_constructor) for item_i in items]
    yield contents_out

@pytest.fixture(scope="function", params=utils.unpack_test_cases(STD_SHAPES), ids=utils.unpack_test_ids(STD_SHAPES))
def std_shapes(request):
    yield deepcopy(request.param)

@pytest.fixture(scope="function", params=utils.unpack_test_cases(STD_SHAPES), ids=utils.unpack_test_ids(STD_SHAPES))
def std_contents(request):
    array_constructor = request.cls.array_constructor
    yield create_contents(request.param, array_constructor)

@pytest.fixture(scope="function", params=utils.unpack_test_cases(STD_SHAPES), ids=utils.unpack_test_ids(STD_SHAPES))
def std_contents_and_shapes(request):
    array_constructor = request.cls.array_constructor
    contents = create_contents(request.param, array_constructor)
    # Note - need to pass in deepcopy of shapes to ensure independence of tests:
    shapes = deepcopy(request.param)
    yield (contents, shapes)

BOOL_VALS = {'true': 1, 'false': 0, 'true&false': None}
BOOL_SHAPES = {f'{key}_{bool_keys}': tuple(product(val, [bool_vals])) for key, val in STD_SHAPES.items() 
                                                                      for bool_keys, bool_vals in BOOL_VALS.items()}
@pytest.fixture(scope="function", params=utils.unpack_test_cases(BOOL_SHAPES), ids=utils.unpack_test_ids(BOOL_SHAPES))
def bool_contents(request):
    shapes, single_value = request.param
    array_constructor = request.cls.array_constructor
    if single_value is not None:
        contents = create_contents(shapes, array_constructor, single_value=single_value, return_bool=True)
    else:
        contents = create_contents(shapes, array_constructor, return_bool=True)
    yield contents