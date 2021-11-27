import jax.numpy as jnp
from arraytainers import Jaxtainer

from test_class import ArraytainerTests

class TestJaxtainer(ArraytainerTests):
    
    container_class = Jaxtainer
    array = jnp.array

    # def test_vmap():

    # def test_jit():

    # def test_autograd():

    # def test_vmap_jit_autograd():



