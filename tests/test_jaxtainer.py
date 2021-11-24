import jax.numpy as jnp
from arraytainers import Jaxtainer

from .test_class import ArraytainerTests

class JaxtainerTests(ArraytainerTests):
    
    self.container_class = Jaxtainer
    self.array = jnp.array

    # def test_vmap():

    # def test_jit():

    # def test_autograd():

    # def test_vmap_jit_autograd():



