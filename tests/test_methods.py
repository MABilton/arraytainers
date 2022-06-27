import jax.numpy as jnp
import numpy as np
import pytest

import helpers
from arraytainers import Arraytainer, Jaxtainer


def test_arraytainer_shallow_copy():
    original_val = np.array(1)
    new_val = np.array(2)
    contents = {"a": [original_val], "b": original_val}

    arraytainer = Arraytainer(contents)
    arraytainer_copy = arraytainer.copy()
    arraytainer_copy["a"][0] = new_val
    arraytainer_copy["b"] = new_val
    arraytainer_copy["c"] = new_val

    assert np.array_equal(arraytainer["a"][0], new_val)
    assert np.array_equal(arraytainer["b"], original_val)
    assert "c" not in arraytainer.keys()


def test_jaxtainer_shallow_copy():
    original_val = jnp.array(1)
    new_val = jnp.array(2)
    contents = {"a": [original_val], "b": original_val}

    jaxtainer = Jaxtainer(contents)
    jaxtainer_copy = jaxtainer.copy()
    jaxtainer_copy["a"][0] = new_val
    jaxtainer_copy["b"] = new_val
    jaxtainer_copy["c"] = new_val

    assert np.array_equal(jaxtainer["a"][0], new_val)
    assert isinstance(jaxtainer["a"][0], jnp.ndarray)
    assert np.array_equal(jaxtainer["b"], original_val)
    assert isinstance(jaxtainer["b"], jnp.ndarray)
    assert "c" not in jaxtainer.keys()


def test_arraytainer_deep_copy():
    original_val = np.array(1)
    new_val = np.array(2)
    contents = {"a": [original_val], "b": original_val}

    arraytainer = Arraytainer(contents)
    arraytainer_copy = arraytainer.deepcopy()
    arraytainer_copy["a"][0] = new_val
    arraytainer_copy["b"] = new_val
    arraytainer_copy["c"] = new_val

    assert np.array_equal(arraytainer["a"][0], original_val)
    assert np.array_equal(arraytainer["b"], original_val)
    assert "c" not in arraytainer.keys()


def test_jaxtainer_deep_copy():
    original_val = jnp.array(1)
    new_val = jnp.array(2)
    contents = {"a": [original_val], "b": original_val}

    jaxtainer = Jaxtainer(contents)
    jaxtainer_copy = jaxtainer.deepcopy()
    jaxtainer_copy["a"][0] = new_val
    jaxtainer_copy["b"] = new_val
    jaxtainer_copy["c"] = new_val

    assert isinstance(jaxtainer["a"][0], jnp.ndarray)
    assert np.array_equal(jaxtainer["a"][0], original_val)
    assert isinstance(jaxtainer["b"], jnp.ndarray)
    assert np.array_equal(jaxtainer["b"], original_val)
    assert "c" not in jaxtainer.keys()


update_dict_tests = [
    {
        "contents": {"a": np.array(1), "b": np.array(2)},
        "key_iterable": None,
        "new_val": {"c": np.array(3)},
        "expected": {"a": np.array(1), "b": np.array(2), "c": np.array(3)},
    },
    {
        "contents": {"a": np.array(1), "b": np.array(2)},
        "key_iterable": None,
        "new_val": {"c": [{"d": np.array(3)}]},
        "expected": {"a": np.array(1), "b": np.array(2), "c": [{"d": np.array(3)}]},
    },
    {
        "contents": [{"a": np.array(1), "b": np.array(2)}],
        "key_iterable": (0,),
        "new_val": {"c": np.array(3)},
        "expected": [{"a": np.array(1), "b": np.array(2), "c": np.array(3)}],
    },
    {
        "contents": {"a": [[{"d": np.array(1)}]], "b": np.array(2)},
        "key_iterable": ("a", 0, 0),
        "new_val": {"c": np.array(3)},
        "expected": {"a": [[{"d": np.array(1), "c": np.array(3)}]], "b": np.array(2)},
    },
    {
        "contents": {"a": [[{"d": np.array(1)}]], "b": np.array(2)},
        "key_iterable": ("a", 0, 0),
        "new_val": {"c": [{"d": np.array(3)}]},
        "expected": {
            "a": [[{"d": np.array(1), "c": [{"d": np.array(3)}]}]],
            "b": np.array(2),
        },
    },
]


@pytest.mark.parametrize(
    "contents, key_iterable, new_val, expected",
    [
        (test["contents"], test["key_iterable"], test["new_val"], test["expected"])
        for test in update_dict_tests
    ],
)
def test_arraytainer_update_with_nonarraytainer(
    contents, key_iterable, new_val, expected
):
    contents = helpers.deepcopy_contents(contents)
    expected = helpers.deepcopy_contents(expected)
    new_val = helpers.deepcopy_contents(new_val)

    arraytainer = Arraytainer(contents)
    if key_iterable is None:
        arraytainer.update(new_val)
    else:
        arraytainer.update(new_val, *key_iterable)

    helpers.assert_equal(arraytainer.unpack(), expected)
    helpers.assert_equal(arraytainer, Arraytainer(expected))


@pytest.mark.parametrize(
    "contents, key_iterable, new_val, expected",
    [
        (test["contents"], test["key_iterable"], test["new_val"], test["expected"])
        for test in update_dict_tests
    ],
)
def test_jaxtainer_update_with_nonjaxtainer(contents, key_iterable, new_val, expected):
    contents = helpers.deepcopy_contents(contents, has_jax_arrays=True)
    expected = helpers.deepcopy_contents(expected, has_jax_arrays=True)
    new_val = helpers.deepcopy_contents(new_val, has_jax_arrays=True)

    jaxtainer = Jaxtainer(contents)
    if key_iterable is None:
        jaxtainer.update(new_val)
    else:
        jaxtainer.update(new_val, *key_iterable)

    helpers.assert_equal(jaxtainer.unpack(), expected)
    helpers.assert_equal(jaxtainer, Jaxtainer(expected))


@pytest.mark.parametrize(
    "contents, key_iterable, new_val, expected",
    [
        (test["contents"], test["key_iterable"], test["new_val"], test["expected"])
        for test in update_dict_tests
    ],
)
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

    helpers.assert_equal(arraytainer.unpack(), expected)
    helpers.assert_equal(arraytainer, Arraytainer(expected))


@pytest.mark.parametrize(
    "contents, key_iterable, new_val, expected",
    [
        (test["contents"], test["key_iterable"], test["new_val"], test["expected"])
        for test in update_dict_tests
    ],
)
def test_jaxtainer_update_with_jaxtainer(contents, key_iterable, new_val, expected):
    contents = helpers.deepcopy_contents(contents, has_jax_arrays=True)
    expected = helpers.deepcopy_contents(expected, has_jax_arrays=True)
    new_val = helpers.deepcopy_contents(new_val, has_jax_arrays=True)

    jaxtainer = Jaxtainer(contents)
    new_val_jaxtainer = Jaxtainer(new_val)
    if key_iterable is None:
        jaxtainer.update(new_val_jaxtainer)
    else:
        jaxtainer.update(new_val_jaxtainer, *key_iterable)

    helpers.assert_equal(jaxtainer.unpack(), expected)
    helpers.assert_equal(jaxtainer, Jaxtainer(expected))


append_list_tests_vals = [
    {
        "contents": [np.array(1), np.array(2)],
        "key_iterable": None,
        "new_val": np.array(3),
        "expected": [np.array(1), np.array(2), np.array(3)],
    },
    {
        "contents": [np.array(1), np.array(2)],
        "key_iterable": None,
        "new_val": [{"d": np.array(3)}],
        "expected": [np.array(1), np.array(2), [{"d": np.array(3)}]],
    },
    {
        "contents": {"a": [np.array(1), np.array(2)]},
        "key_iterable": ("a",),
        "new_val": {"c": np.array(3)},
        "expected": {"a": [np.array(1), np.array(2), {"c": np.array(3)}]},
    },
    {
        "contents": [{"a": [np.array(1)], "b": np.array(2)}],
        "key_iterable": (0, "a"),
        "new_val": np.array(3),
        "expected": [{"a": [np.array(1), np.array(3)], "b": np.array(2)}],
    },
    {
        "contents": {"a": [[{"d": np.array(1)}]], "b": np.array(2)},
        "key_iterable": ("a", 0),
        "new_val": {"c": [{"d": np.array(3)}]},
        "expected": {
            "a": [[{"d": np.array(1)}, {"c": [{"d": np.array(3)}]}]],
            "b": np.array(2),
        },
    },
]


@pytest.mark.parametrize(
    "contents, key_iterable, new_val, expected",
    [
        (test["contents"], test["key_iterable"], test["new_val"], test["expected"])
        for test in append_list_tests_vals
    ],
)
def test_arraytainer_append_with_nonarraytainer(
    contents, key_iterable, new_val, expected
):
    contents = helpers.deepcopy_contents(contents)
    expected = helpers.deepcopy_contents(expected)
    new_val = helpers.deepcopy_contents(new_val)

    arraytainer = Arraytainer(contents)
    if key_iterable is None:
        arraytainer.append(new_val)
    else:
        arraytainer.append(new_val, *key_iterable)

    helpers.assert_equal(arraytainer.unpack(), expected)
    helpers.assert_equal(arraytainer, Arraytainer(expected))


@pytest.mark.parametrize(
    "contents, key_iterable, new_val, expected",
    [
        (test["contents"], test["key_iterable"], test["new_val"], test["expected"])
        for test in append_list_tests_vals
    ],
)
def test_jaxtainer_append_with_nonjaxtainer(contents, key_iterable, new_val, expected):
    contents = helpers.deepcopy_contents(contents, has_jax_arrays=True)
    expected = helpers.deepcopy_contents(expected, has_jax_arrays=True)
    new_val = helpers.deepcopy_contents(new_val, has_jax_arrays=True)

    jaxtainer = Jaxtainer(contents)
    if key_iterable is None:
        jaxtainer.append(new_val)
    else:
        jaxtainer.append(new_val, *key_iterable)

    helpers.assert_equal(jaxtainer.unpack(), expected)
    helpers.assert_equal(jaxtainer, Jaxtainer(expected))


append_list_tests_arraytainers = [
    {
        "contents": [np.array(1), np.array(2)],
        "key_iterable": None,
        "new_val": [np.array(3)],
        "expected": [np.array(1), np.array(2), [np.array(3)]],
    },
    {
        "contents": [np.array(1), np.array(2)],
        "key_iterable": None,
        "new_val": [{"d": np.array(3)}],
        "expected": [np.array(1), np.array(2), [{"d": np.array(3)}]],
    },
    {
        "contents": {"a": [np.array(1), np.array(2)]},
        "key_iterable": ("a",),
        "new_val": {"c": np.array(3)},
        "expected": {"a": [np.array(1), np.array(2), {"c": np.array(3)}]},
    },
    {
        "contents": [{"a": [np.array(1)], "b": np.array(2)}],
        "key_iterable": (0, "a"),
        "new_val": {"c": np.array(3)},
        "expected": [{"a": [np.array(1), {"c": np.array(3)}], "b": np.array(2)}],
    },
    {
        "contents": {"a": [[{"d": np.array(1)}]], "b": np.array(2)},
        "key_iterable": ("a", 0),
        "new_val": {"c": [{"d": np.array(3)}]},
        "expected": {
            "a": [[{"d": np.array(1)}, {"c": [{"d": np.array(3)}]}]],
            "b": np.array(2),
        },
    },
]


@pytest.mark.parametrize(
    "contents, key_iterable, new_val, expected",
    [
        (test["contents"], test["key_iterable"], test["new_val"], test["expected"])
        for test in append_list_tests_arraytainers
    ],
)
def test_arraytainer_append_with_arraytainer(contents, key_iterable, new_val, expected):
    contents = helpers.deepcopy_contents(contents)
    expected = helpers.deepcopy_contents(expected)
    new_val = helpers.deepcopy_contents(new_val)

    arraytainer = Arraytainer(contents)
    new_val_arraytainer = Arraytainer(new_val)
    if key_iterable is None:
        arraytainer.append(new_val_arraytainer)
    else:
        arraytainer.append(new_val_arraytainer, *key_iterable)

    helpers.assert_equal(arraytainer.unpack(), expected)
    helpers.assert_equal(arraytainer, Arraytainer(expected))


@pytest.mark.parametrize(
    "contents, key_iterable, new_val, expected",
    [
        (test["contents"], test["key_iterable"], test["new_val"], test["expected"])
        for test in append_list_tests_arraytainers
    ],
)
def test_jaxtainer_append_with_jaxtainer(contents, key_iterable, new_val, expected):
    contents = helpers.deepcopy_contents(contents, has_jax_arrays=True)
    expected = helpers.deepcopy_contents(expected, has_jax_arrays=True)
    new_val = helpers.deepcopy_contents(new_val, has_jax_arrays=True)

    jaxtainer = Jaxtainer(contents)
    new_val_jaxtainer = Jaxtainer(new_val)
    if key_iterable is None:
        jaxtainer.append(new_val_jaxtainer)
    else:
        jaxtainer.append(new_val_jaxtainer, *key_iterable)

    helpers.assert_equal(jaxtainer.unpack(), expected)
    helpers.assert_equal(jaxtainer, Jaxtainer(expected))


append_dict_error_tests = [
    {"contents": {"a": np.array(1)}, "key_iterable": None},
    {"contents": [{"a": np.array(1)}], "key_iterable": (0,)},
    {"contents": {"b": [{"a": np.array(1)}]}, "key_iterable": ("b", 0)},
]


@pytest.mark.parametrize(
    "contents, key_iterable",
    [(test["contents"], test["key_iterable"]) for test in append_dict_error_tests],
)
def test_arraytainer_append_to_dict_error(contents, key_iterable):
    new_val = np.array(1)
    contents = helpers.deepcopy_contents(contents)
    arraytainer = Arraytainer(contents)

    with pytest.raises(TypeError, match="Can't append to dictionary-like Arraytainer"):
        if key_iterable is None:
            arraytainer.append(new_val)
        else:
            arraytainer.append(new_val, *key_iterable)


update_list_error_tests = [
    {"contents": [np.array(1)], "key_iterable": None},
    {"contents": {"a": [np.array(1)]}, "key_iterable": ("a",)},
    {"contents": [{"a": [np.array(1)]}], "key_iterable": (0, "a")},
]


@pytest.mark.parametrize(
    "contents, key_iterable",
    [(test["contents"], test["key_iterable"]) for test in update_list_error_tests],
)
def test_arraytainer_update_list_error(contents, key_iterable):
    new_val = {123: np.array(1)}
    contents = helpers.deepcopy_contents(contents)
    arraytainer = Arraytainer(contents)

    with pytest.raises(TypeError, match="Can't update a list-like Arraytainer"):
        if key_iterable is None:
            arraytainer.update(new_val)
        else:
            arraytainer.update(new_val, *key_iterable)


too_many_keys_tests = [
    {"contents": {"a": [np.array(1)]}, "key_iterable": ("a", 0)},
    {"contents": [{"a": np.array(1)}], "key_iterable": (0, "a")},
    {"contents": {"a": [np.array(1)]}, "key_iterable": ("a", 0, 0)},
    {"contents": [{"a": np.array(1)}], "key_iterable": (0, "a", "a")},
]


@pytest.mark.parametrize(
    "contents, key_iterable",
    [(test["contents"], test["key_iterable"]) for test in too_many_keys_tests],
)
def test_arraytainer_update_error_too_many_keys_provided(contents, key_iterable):
    new_val = {"a": np.array(1)}
    contents = helpers.deepcopy_contents(contents)
    arraytainer = Arraytainer(contents)

    with pytest.raises(
        KeyError, match="(not an Arraytainer)|(not a key in this Arraytainer)"
    ):
        arraytainer.update(new_val, key_iterable)


@pytest.mark.parametrize(
    "contents, key_iterable",
    [(test["contents"], test["key_iterable"]) for test in too_many_keys_tests],
)
def test_arraytainer_append_list_too_many_keys_provided(contents, key_iterable):
    new_val = np.array(1)
    contents = helpers.deepcopy_contents(contents)
    arraytainer = Arraytainer(contents)

    with pytest.raises(
        KeyError, match="(not an Arraytainer)|(not a key in this Arraytainer)"
    ):
        arraytainer.append(new_val, key_iterable)


@pytest.mark.parametrize(
    "contents, key_iterable",
    [(test["contents"], test["key_iterable"]) for test in too_many_keys_tests],
)
def test_jaxtainer_update_error_too_many_keys_provided(contents, key_iterable):
    new_val = {"a": jnp.array(1)}
    contents = helpers.deepcopy_contents(contents, has_jax_arrays=True)
    jaxtainer = Jaxtainer(contents)

    with pytest.raises(
        KeyError, match="(not an Arraytainer)|(not a key in this Arraytainer)"
    ):
        jaxtainer.update(new_val, key_iterable)


@pytest.mark.parametrize(
    "contents, key_iterable",
    [(test["contents"], test["key_iterable"]) for test in too_many_keys_tests],
)
def test_jaxtainer_append_list_too_many_keys_provided(contents, key_iterable):
    new_val = jnp.array(1)
    contents = helpers.deepcopy_contents(contents, has_jax_arrays=True)
    jaxtainer = Jaxtainer(contents)

    with pytest.raises(
        KeyError, match="(not an Arraytainer)|(not a key in this Arraytainer)"
    ):
        jaxtainer.append(new_val, key_iterable)


unpack_tests = [
    [np.array(1), np.array(2)],
    {"a": np.array(1), "b": np.array(2)},
    {"a": [np.array(1)], "b": {"a": [np.array(2)], "b": {"a": np.array(3)}}},
    [{"a": {"a": np.array(1)}, "b": [np.array(2)]}, [[np.array(3)], np.array(4)]],
]


@pytest.mark.parametrize("contents", unpack_tests)
def test_arraytainer_unpack(contents):
    contents = helpers.deepcopy_contents(contents)
    arraytainer = Arraytainer(contents)
    unpacked = arraytainer.unpack()
    helpers.assert_equal(unpacked, contents)


@pytest.mark.parametrize("contents", unpack_tests)
def test_jaxtainer_unpack(contents):
    contents = helpers.deepcopy_contents(contents, has_jax_arrays=True)
    jaxtainer = Jaxtainer(contents)
    unpacked = jaxtainer.unpack()
    helpers.assert_equal(unpacked, contents)


tolist_tests = [
    {
        "contents": {"a": np.array([[1, 2]]), 1: np.array(1)},
        "expected": {"a": ([1, 2],), 1: 1},
    },
    {
        "contents": {"a": [np.array([[1, 2]])], 1: [np.array(1)]},
        "expected": {"a": [([1, 2],)], 1: [1]},
    },
    {
        "contents": [[{"a": np.array([[[1, 2]]]), "b": np.array(3)}], [np.array([4])]],
        "expected": [[{"a": ([[1, 2]],), "b": 3}], [(4,)]],
    },
    {
        "contents": [
            [{"a": [np.array([[[1, 2]]])], "b": [np.array(3)]}],
            [np.array([4])],
        ],
        "expected": [[{"a": [([[1, 2]],)], "b": [3]}], [(4,)]],
    },
]


@pytest.mark.parametrize(
    "contents, expected",
    [(test["contents"], test["expected"]) for test in tolist_tests],
)
def test_arraytainer_tolist(contents, expected):
    contents = helpers.deepcopy_contents(contents)

    arraytainer = Arraytainer(contents)
    output = arraytainer.tolist()

    helpers.assert_equal(output, expected)
    helpers.assert_equal(arraytainer, Arraytainer(output))


@pytest.mark.parametrize(
    "contents, expected",
    [(test["contents"], test["expected"]) for test in tolist_tests],
)
def test_jaxtainer_tolist(contents, expected):
    contents = helpers.deepcopy_contents(contents, has_jax_arrays=True)

    jaxtainer = Jaxtainer(contents)
    output = jaxtainer.tolist()

    helpers.assert_equal(output, expected)
    helpers.assert_equal(jaxtainer, Jaxtainer(output))
