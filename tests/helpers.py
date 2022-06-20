from arraytainers import Arraytainer
import numpy as np
import jax.numpy as jnp
import copy

def assert_equal(output, expected, approx_equal=False):
    if isinstance(expected, (list, dict, Arraytainer)):
        assert_contents_equal(output, expected, approx_equal)
    else:
        assert_arrays_equal(output, expected, approx_equal)

def assert_arrays_equal(output, expected, approx_equal=False):
    assert isinstance(output, (np.ndarray, jnp.ndarray))
    assert isinstance(expected, (np.ndarray, jnp.ndarray))
    assert type(output) == type(expected)
    assert output.dtype == expected.dtype
    if approx_equal:
        assert output.shape == expected.shape
        assert np.allclose(output, expected)
    else:
        assert np.array_equal(output, expected)

def assert_contents_equal(output, expected, approx_equal=False):
    assert isinstance(output, (Arraytainer, list, dict))
    assert isinstance(expected, (Arraytainer, list, dict))
    output_keys = range(len(output)) if isinstance(output, list) else output.keys()
    expected_keys = range(len(expected)) if isinstance(expected, list) else expected.keys()
    assert output_keys == expected_keys
    for key in expected_keys:
        assert_equal(output[key], expected[key], approx_equal)

def deepcopy_contents(contents, dtype=None, has_jax_arrays=False):
    contents = copy.deepcopy(contents)
    if has_jax_arrays:
        contents = convert_arrays_to_jax(contents, dtype)
    elif dtype is not None:
        contents = convert_arrays_to_numpy(contents, dtype)
    return contents

def convert_arrays_to_jax(contents, dtype=None): 
    if isinstance(contents, (np.ndarray, jnp.ndarray)):
        contents = jnp.array(contents, dtype=dtype)
    elif isinstance(contents, (dict, list, Arraytainer)):
        items = enumerate(contents) if isinstance(contents, list) else contents.items()
        for key, val in items:
            if isinstance(val, (np.ndarray, jnp.ndarray)):
                contents[key] = jnp.array(val, dtype=dtype)
            else:
                contents[key] = convert_arrays_to_jax(val, dtype)
    return contents

def convert_arrays_to_numpy(contents, dtype=None): 
    if isinstance(contents, (np.ndarray, jnp.ndarray)):
        contents = np.array(contents, dtype=dtype)
    elif isinstance(contents, (dict, list, Arraytainer)):
        items = enumerate(contents) if isinstance(contents, list) else contents.items()
        for key, val in items:
            if isinstance(val, (np.ndarray, jnp.ndarray)):
                contents[key] = np.array(val, dtype=dtype)
            else:
                contents[key] = convert_arrays_to_numpy(val, dtype)
    return contents