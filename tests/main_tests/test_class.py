import pytest
from jax.core import InconclusiveDimensionOperation

from .test_array_methods import ArrayMixin
from .test_functions import FunctionMixin
from .test_indexing import IndexMixin
from .test_iterables import IterableMixin

class ArraytainerTests(ArrayMixin, FunctionMixin, IndexMixin, IterableMixin):

    # Attributes which have different values for Jax and Numpy tests:
    container_class = None
    array = None
    array_types = None

    # Convenience method to assert that a particular function call throws a particular exception:
    def assert_exception(self, function, exception, *args, **kwargs):
        
        if not isinstance(exception, Exception):
            # Group errors according to 'functionally equivalent' groupings:
            testing_numpytainer = 'numpy' in self.container_class.__name__.lower()
            error_groupings = {'key': (IndexError, KeyError, TypeError), # Thrown by indexing/key errors
                               'broadcast': (ValueError,) if testing_numpytainer else (TypeError,) # Thrown by broadcasting errors
                               'reshape': (ValueError, InconclusiveDimensionOperation)}
            for key, group in error_groupings.items(): 
                if key in exception.lower():
                    exception_class = group
                    break
        else:
            exception_class = exception.__class__
        
        with pytest.raises(exception_class):
            function(*args, **kwargs)