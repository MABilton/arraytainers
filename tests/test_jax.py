import pytest
import jax
import jax.numpy
from arraytainers import Jaxtainer


def computations(array):
    y = np.sin(x**2 + 1)
    y = y.reshape(1,-1)
    y = y.squeeze()
    z = y.size
    y.sum()
    

def jax_test_func(array):
    x = Jaxtainer()
    
    return 

def test_vmap():
    pass

def test_jacfwd():
    pass

def test_vmap_and_jacfwd():
    pass


def test_jaxtainer_deepcopy_no_conversion_to_numpy_array():
    pass