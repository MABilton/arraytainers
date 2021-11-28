import pytest
import numpy as np

from .utils import apply_func_to_contents, assert_equal_values, assert_same_types

SINGLE_ARG_FUNCS = \
{'exp': np.exp, 'cos': np.cos, 'floor': np.floor, 'log': np.log, 'transpose': lambda x: x.T,
'add_1': lambda x: x+2, 'exponent_1': lambda x: x**2, 'multiply': lambda x: x*2, 'floor_div_1': lambda x: x//2,
'subtract': lambda x: 2-x, 'division': lambda x: 2/x, 'exponent_2': lambda x: 2**x, 'floor_div_2': lambda x: 2//x,
'matmult_1': lambda x: 0.5*np.ones(1) @ x, 'matmult_2': lambda x: x @ 0.5*np.ones(1)}
    
class FunctionMixin:
    
    @pytest.mark.parametrize('single_arg_func', SINGLE_ARG_FUNCS.values(), ids=SINGLE_ARG_FUNCS.keys())
    def test_apply_function_single_arg(self, std_contents, single_arg_func):
        
        arraytainer = self.container_class(std_contents)

        expected, exception = apply_func_to_contents(std_contents, func=single_arg_func)
        if exception is None:
            result = single_arg_func(arraytainer)
            assert_equal_values(result.unpacked, expected, approx_equal=True)
            assert_same_types(result, self.container_class(expected))
        else:
            # Jax is inconsistent in terms of the errors it throws for broadcasting errors - 
            # easier just to check for an exception:
            self.assert_exception(single_arg_func, Exception(), arraytainer)
    
    # @pytest.mark.parametrize('contents_list, exception', 
    # [],
    # indirect=[contents]
    # )
    # def test_apply_function_solve(self, contents_list, exception):
        
    #     arraytainers = [self.container_class(content_i) for content_i in contents_list]

    #     with exception:
    #         result = func(*arraytainers)

    #     expected = apply_func_to_contents(*contents_list, func=func)

    #     assert assert_equal(result.unpacked, expected, approx_equal=True)

    # def test_apply_function_einsum(self, contents_list, func, exception):
        
    #     arraytainers = [self.container_class(content_i) for content_i in contents_list]

    #     with exception:
    #         result = func(*arraytainers)

    #     expected = apply_func_to_contents(*contents_list, func=func)

    #     assert assert_equal(result.unpacked, expected, approx_equal=True)

    # def test_apply_function_kron(self, contents_list, func, exception):
        
    #     arraytainers = [self.container_class(content_i) for content_i in contents_list]

    #     with exception:
    #         result = func(*arraytainers)

    #     expected = apply_func_to_contents(*contents_list, func=func)

    #     assert assert_equal(result.unpacked, expected, approx_equal=True)