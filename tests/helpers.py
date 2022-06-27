import copy
from numbers import Number

import jax.numpy as jnp
import numpy as np

from arraytainers import Arraytainer


def assert_equal(output, expected, approx_equal=False):
    if isinstance(expected, (list, dict, tuple, Arraytainer)):
        assert_contents_equal(output, expected, approx_equal)
    elif isinstance(expected, (np.ndarray, jnp.ndarray)):
        assert_arrays_equal(output, expected, approx_equal)
    else:
        assert_numbers_equal(output, expected)


def assert_numbers_equal(output, expected, approx_equal=False):
    assert isinstance(output, Number)
    assert isinstance(expected, Number)
    if approx_equal:
        assert np.allclose(output, expected)
    else:
        assert output == expected


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
    assert isinstance(output, (list, dict, tuple, Arraytainer))
    assert isinstance(expected, (list, dict, tuple, Arraytainer))
    assert type(output) == type(expected)
    output_keys = (
        range(len(output)) if isinstance(output, (list, tuple)) else output.keys()
    )
    expected_keys = (
        range(len(expected)) if isinstance(expected, (list, tuple)) else expected.keys()
    )
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
