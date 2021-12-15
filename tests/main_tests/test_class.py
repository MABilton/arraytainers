import pytest
from jax.core import InconclusiveDimensionOperation

from .test_array_methods import ArrayMixin
from .test_functions import FunctionMixin
from .test_indexing import IndexMixin
from .test_iterables import IterableMixin
from .test_setting import SetMixin

class ArraytainerTests(ArrayMixin, FunctionMixin, IndexMixin, IterableMixin, SetMixin): 

    # Attributes which have different values for Jax and Numpy testing classes:
    container_class = None
    array_class = None

    # Convenience method to assert that a particular function call throws a particular exception:
    def assert_exception(self, function, exception, *args, **kwargs):
        # Is exception is a string:
        if isinstance(exception, str):
            # Group errors according to 'functionally equivalent' groupings:
            testing_numpytainer = 'numpy' in self.container_class.__name__.lower()
            error_groupings = {'key': (IndexError, KeyError, TypeError), # Indexing/key errors
                               'broadcast': (ValueError,) if testing_numpytainer else (TypeError, ValueError), # Broadcasting errors
                               'reshape': (ValueError,) if testing_numpytainer else (InconclusiveDimensionOperation,)} # Reshaping error
            for key, group in error_groupings.items(): 
                if key in exception.lower():
                    exception_class = group
                    break
        # If exception is an exception type (e.g. ValueError):
        elif isinstance(exception, type):
            exception_class = (exception,)
        # If exception is an exception object which has been thrown:
        else:
            exception_class = exception.__class__
        
        with pytest.raises(exception_class):
            function(*args, **kwargs)