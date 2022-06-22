import pytest
from arraytainers import Arraytainer, Jaxtainer
import numpy as np
import jax.numpy as jnp

import helpers

def test_arraytainer_shallow_copy():
    original_val = np.array(1)
    new_val =  np.array(2)
    contents = {'a': [original_val], 'b': original_val}

    arraytainer = Arraytainer(contents)
    arraytainer_copy = arraytainer.copy()
    arraytainer_copy['a'][0] = new_val
    arraytainer_copy['b'] = new_val
    arraytainer_copy['c'] = new_val

    assert np.array_equal(arraytainer['a'][0], new_val)
    assert np.array_equal(arraytainer['b'], original_val)
    assert 'c' not in arraytainer.keys()

def test_jaxtainer_shallow_copy():
    original_val = jnp.array(1)
    new_val =  jnp.array(2)
    contents = {'a': [original_val], 'b': original_val}

    jaxtainer = Jaxtainer(contents)
    jaxtainer_copy = jaxtainer.copy()
    jaxtainer_copy['a'][0] = new_val
    jaxtainer_copy['b'] = new_val
    jaxtainer_copy['c'] = new_val

    assert np.array_equal(jaxtainer['a'][0], new_val)
    assert isinstance(jaxtainer['a'][0], jnp.ndarray)
    assert np.array_equal(jaxtainer['b'], original_val)
    assert isinstance(jaxtainer['b'], jnp.ndarray)
    assert 'c' not in jaxtainer.keys()

def test_arraytainer_deep_copy():
    original_val = np.array(1)
    new_val =  np.array(2)
    contents = {'a': [original_val], 'b': original_val}

    arraytainer = Arraytainer(contents)
    arraytainer_copy = arraytainer.deepcopy()
    arraytainer_copy['a'][0] = new_val
    arraytainer_copy['b'] = new_val
    arraytainer_copy['c'] = new_val

    assert np.array_equal(arraytainer['a'][0], original_val)
    assert np.array_equal(arraytainer['b'], original_val)
    assert 'c' not in arraytainer.keys()

def test_jaxtainer_deep_copy():
    original_val = jnp.array(1)
    new_val =  jnp.array(2)
    contents = {'a': [original_val], 'b': original_val}

    jaxtainer = Jaxtainer(contents)
    jaxtainer_copy = jaxtainer.deepcopy()
    jaxtainer_copy['a'][0] = new_val
    jaxtainer_copy['b'] = new_val
    jaxtainer_copy['c'] = new_val

    assert isinstance(jaxtainer['a'][0], jnp.ndarray)
    assert np.array_equal(jaxtainer['a'][0], original_val)
    assert isinstance(jaxtainer['b'], jnp.ndarray)
    assert np.array_equal(jaxtainer['b'], original_val)
    assert 'c' not in jaxtainer.keys()

update_dict_tests = [
    {'contents': {'a':np.array(1), 'b':np.array(2)},
     'key_iterable': None,
     'new_val': {'c': np.array(3)},
     'expected': {'a':np.array(1), 'b':np.array(2), 'c':np.array(3)}},
    {'contents': {'a':np.array(1), 'b':np.array(2)},
     'key_iterable': None,
     'new_val': {'c': [{'d':np.array(3)}]},
     'expected': {'a':np.array(1), 'b':np.array(2), 'c':[{'d':np.array(3)}]}},
    {'contents': [{'a':np.array(1), 'b':np.array(2)}],
     'key_iterable': (0,),
     'new_val': {'c': np.array(3)},
     'expected': [{'a':np.array(1), 'b':np.array(2), 'c':np.array(3)}]},
    {'contents': {'a':[[{'d':np.array(1)}]], 'b':np.array(2)},
     'key_iterable': ('a',0,0),
     'new_val': {'c': np.array(3)},
     'expected': {'a':[[{'d':np.array(1), 'c':np.array(3)}]], 'b':np.array(2)} },
    {'contents': {'a':[[{'d':np.array(1)}]], 'b':np.array(2)},
     'key_iterable': ('a',0,0),
     'new_val': {'c': [{'d':np.array(3)}]},
     'expected': {'a':[[{'d':np.array(1), 'c':[{'d':np.array(3)}]}]], 'b':np.array(2)} },
]
@pytest.mark.parametrize("contents, key_iterable, new_val, expected", [(test['contents'], test['key_iterable'], test['new_val'], test['expected']) for test in update_dict_tests])
def test_arraytainer_update_with_nonarraytainer(contents, key_iterable, new_val, expected):
    contents = helpers.deepcopy_contents(contents)
    expected = helpers.deepcopy_contents(expected)
    new_val = helpers.deepcopy_contents(new_val)
    
    arraytainer = Arraytainer(contents)
    if key_iterable is None:
        arraytainer.update(new_val)
    else:
        arraytainer.update(new_val, *key_iterable)

    helpers.assert_equal(arraytainer.unpack(),expected)
    helpers.assert_equal(arraytainer, Arraytainer(expected))

@pytest.mark.parametrize("contents, key_iterable, new_val, expected", [(test['contents'], test['key_iterable'], test['new_val'], test['expected']) for test in update_dict_tests])
def test_arraytainer_update_with_arraytainer(contents, key_iterable, new_val, expected):
    contents = helpers.deepcopy_contents(contents)
    expected = helpers.deepcopy_contents(expected)
    new_val = helpers.deepcopy_contents(new_val)

    arraytainer = Arraytainer(contents)
    new_val_arraytainer = Arraytainer(new_val)
    if key_iterable is None:
        arraytainer.update(new_val_arraytainer)
    else:
        arraytainer.update(new_val_arraytainer, *key_iterable)

    helpers.assert_equal(arraytainer.unpack(),expected)
    helpers.assert_equal(arraytainer, Arraytainer(expected))

def test_arraytainer_append():
    pass

append_dict_error_tests = [
    {'contents': {'a': np.array(1)},
    'key_iterable': None},
    {'contents': [{'a': np.array(1)}],
    'key_iterable': (0,)},
    {'contents': {'b':[{'a': np.array(1)}]},
    'key_iterable': ('b',0)},
]
@pytest.mark.parametrize("contents, key_iterable", [(test['contents'], test['key_iterable']) for test in append_dict_error_tests])
def test_arraytainer_append_to_dict_error(contents, key_iterable):
    new_val = np.array(1)
    contents = helpers.deepcopy_contents(contents)
    arraytainer = Arraytainer(contents)
    if key_iterable is None:
        key_iterable = []
    with pytest.raises(TypeError, match="Can't append to dictionary-like Arraytainer"):
        if key_iterable:
            arraytainer.append(new_val, *key_iterable)
        else:
            arraytainer.append(new_val)


update_list_error_tests = [
    {'contents': [np.array(1)],
    'key_iterable': None},
    {'contents': {'a': [np.array(1)]},
    'key_iterable': ('a',)},
    {'contents': [{'a':[np.array(1)]}],
    'key_iterable': (0,'a')},
]
@pytest.mark.parametrize("contents, key_iterable", [(test['contents'], test['key_iterable']) for test in update_list_error_tests])
def test_arraytainer_update_list_error(contents, key_iterable):
    new_val = {123: np.array(1)}
    contents = helpers.deepcopy_contents(contents)
    arraytainer = Arraytainer(contents)
    if key_iterable is None:
        key_iterable = []
    with pytest.raises(TypeError, match="Can't update a list-like Arraytainer"):
        if key_iterable:
            arraytainer.update(new_val, *key_iterable)
        else:
            arraytainer.update(new_val)

def test_jaxtainer_update():
    pass

def test_jaxtainer_append():
    pass

def test_jaxtainer_append_to_dict_error():
    pass

def test_jaxtainer_update_list_error():
    pass