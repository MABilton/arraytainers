import jax.numpy as jnp
import jaxlib
from arraytainers import Jaxtainer
from main_tests.test_class import ArraytainerTests

class TestJaxtainer(ArraytainerTests):
    
    container_class = Jaxtainer
    expected_array_types = (jaxlib.xla_extension.DeviceArrayBase,)

    @staticmethod
    def array_constructor(x):
        return jnp.array(object=x)

    # self.array = lambda self, x : jnp.array(object=x)
    # self.array_types = (jaxlib.xla_extension.DeviceArrayBase,)

    # def test_vmap():

    # def test_jit():

    # def test_autograd():

    # def test_vmap_jit_autograd():