import pytest
import jax
import jax.numpy
from arraytainers import Jaxtainer


def jax_test_func(arg1, arg2):
    x = Jaxtainer()
    return x**2 + 1

def test_vmap():
    pass

def test_jacfwd():
    pass

def test_vmap_and_jacfwd():
    pass


def test_jaxtainer_deepcopy_no_conversion_to_numpy_array():
    pass