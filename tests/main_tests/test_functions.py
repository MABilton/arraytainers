import pytest
import numpy as np
from numbers import Number
from . import utils
from .utils import cartesian_prod

class FunctionMixin:
    
    def perform_function_test(self, *contents_list, func=None, args=(), kwargs=None, exception=False):
        kwargs = {} if kwargs is None else kwargs
        arraytainer_list = [self.container_class(contents) for contents in contents_list]
        expected, found_exception = utils.apply_func_to_contents(*contents_list, func=func, args=args, kwargs=kwargs)  
        # If an exception is not specified, set it to exception encountered by apply_func_to_contents; note that
        # None is truthy in Python - this assignment will not occur if specified exception=None:
        if not exception:
            exception = found_exception
        
        if exception is None:
            result = func(*arraytainer_list, *args, **kwargs)
            utils.assert_equal_values(result.unpacked, expected, approx_equal=True)
            utils.assert_same_types(result, self.container_class(expected))
        else:
            self.assert_exception(func, exception, *arraytainer_list)

    # def test_apply_method(self, contents, func, args, kwargs):
    #     pass

    NP_FUNC_TEST_CASES = {'exp': np.exp, 'cos': np.cos, 'floor': np.floor, 'log': np.log}
    @pytest.mark.parametrize('np_func', NP_FUNC_TEST_CASES.values(), ids=NP_FUNC_TEST_CASES.keys())
    def test_np_func(self, std_contents, np_func):
        self.perform_function_test(std_contents, func=np_func)

    UFUNC_TEST_CASES = {'addition': lambda x,y:x+y, 'subtraction': lambda x,y:x-y, 'multiplication': lambda x,y:x*y, 
                        'division': lambda x,y:x/y, 'exponent': lambda x,y:x**y, 'less_than': lambda x,y:x<y, 
                        'greater_than_or_equal_to': lambda x,y:x>=y, 'equality': lambda x,y: x==y}
    @pytest.mark.parametrize('ufunc', UFUNC_TEST_CASES.values(), ids=UFUNC_TEST_CASES.keys())
    @pytest.mark.parametrize('scalar', [2, 2.0], ids=['int', 'float'])
    def test_ufunc_scalar(self, std_contents, scalar, ufunc):
        # Make sure that ufuncs are handled correctly when arraytainer before scalar and vice versa:
        for func_i in (lambda x: ufunc(x, scalar), lambda x: ufunc(scalar, x)):
            self.perform_function_test(std_contents, func=func_i)

    ARRAY_EXCEPTION_TESTS = ( ([2],None), ([2,2],None), ([[2]],None), ([[2, 2]],None), ([2,2,1],ValueError) )
    UFUNC_ARRAY_TEST_CASES = \
    {'list': cartesian_prod( [[(2,2),(1,2)],[(1,2),(2,2)]], ARRAY_EXCEPTION_TESTS ),
     'dict': cartesian_prod( {'a':{'a':(2,2),'b':(1,1)},'b':{'a':(1,1),'b':(2,2)}}, ARRAY_EXCEPTION_TESTS ),
     'mixed_A': cartesian_prod( {'a':[(2,2),(1,2)],'b':[(1,2),(2,2)]} , ARRAY_EXCEPTION_TESTS ),
     'mixed_B': cartesian_prod( [{'a':{'c':(2,2)},'b':(1,1)},{'a':[(1,1),(1,1)],'b':(2,2)}], ARRAY_EXCEPTION_TESTS ) }
    @pytest.mark.parametrize('ufunc', UFUNC_TEST_CASES.values(), ids=UFUNC_TEST_CASES.keys())
    @pytest.mark.parametrize('contents, array_shape, exception', utils.unpack_test_cases(UFUNC_ARRAY_TEST_CASES), 
                                                                 ids=utils.unpack_test_ids(UFUNC_ARRAY_TEST_CASES), 
                                                                 indirect=['contents'])
    def test_ufunc_array(self, contents, array_shape, exception, ufunc):
        # Make sure arraytainers work with Numpy arrays AND Jax arrays:
        for array_constructor in utils.ARRAY_CONSTRUCTORS:
            array = array_constructor(array_shape)
            # Make sure that ufuncs are handled correctly when arraytainer before array and vice versa:
            for func_i in (lambda x: ufunc(x, array), lambda x: ufunc(array, x)):
                self.perform_function_test(contents, func=func_i, exception=exception)

    EINSUM_TEST_CASES = \
    {'list': cartesian_prod( [[(3,3,4),(3,4,5)],(2,2,3)], 
                                ( ( [[(4,5),(5,6)],(3,4)], 'ijk,kl->ijl', None ), 
                                  ( [(2,3),(3,2)], 'ijk,li->ljk', None ), 
                                  ( [[(2,3)],(3,2)], 'ijk,li->ljk', 'key_error' ),
                                  ( [(2,3),(3,3)], 'ijk,li->ljk', ValueError ) ) ),
     'dict': cartesian_prod({'a':(3,3,4),'b':{'a':(3,4,5),'b':(3,2,3)}}, 
                                ( ( {'a':(4,1),'b':{'a':(5,2),'b':(3,1)}}, 'ijk,kl->ijl', None ), 
                                    ( {'a':(4,3),'b':(2,3)}, 'ijk,li->ljk', None ),
                                    ( {'a':(4,3),'c':(2,3)}, 'ijk,li->ljk', 'key_error' ),
                                    ( {'a':(4,1),'b':{'a':(6,2),'b':(3,1)}}, 'ijk,kl->ijl', ValueError ) ) ),
     'mixed': cartesian_prod([{'a':[(2,3,4),(2,3,3)],'b':[(2,3,1),(2,1,2)]},{'c':[(3,2,3),(3,3,2)],'d':[(3,2,2),(3,2,1)]}],    
               ( ( [{'a':[(4,1),(3,1)],'b':[(1,2),(2,3)]},{'c':[(3,2),(2,1)],'d':[(2,2),(1,2)]}], 'ijk,kl->ijl', None ),
                 ( [(3,2),(1,3)], 'ijk,li->ljk', None ), 
                 ( [{'a':(1,2),'b':(3,2)},{'c':(1,3),'d':(1,3)}], 'ijk,li->ljk', None ),
                 ( [{'a':(1,2),'b':(3,2)},{'c':(1,3),'e':(1,3)}], 'ijk,li->ljk', 'key_error' ),
                 ( [{'a':(1,2),'b':(3,2)},{'c':(1,3),'d':(1,4)}], 'ijk,kl->ijl', ValueError ) ) )}
    # Need to wrap contents values in list so that they may be passed onto contents_list fixture:
    EINSUM_TEST_CASES = utils.group_first_n_params(EINSUM_TEST_CASES, n=2)
    @pytest.mark.parametrize('contents_list, einsum_str, exception', utils.unpack_test_cases(EINSUM_TEST_CASES), 
                                                                     ids=utils.unpack_test_ids(EINSUM_TEST_CASES), 
                                                                     indirect=['contents_list'])
    def test_einsum(self, contents_list, einsum_str, exception):
        einsum_func = lambda x,y : np.einsum(einsum_str, x, y)
        self.perform_function_test(*contents_list, func=einsum_func, exception=exception)

    MATMULT_TEST_CASES = \
    {'list': cartesian_prod( [[(2,2),(1,2)],[(2,3),[(1,3),(3,3)]]],  
                        ( ( [(2,3),(3,2)], None ),
                          ( [[(2,3),(2,1)],[(3,1),(3,2)]], None ),
                          ( [[(2,3),(2,1)],[(3,1),[(3,2),(3,1)]]], None ),
                          ( [[(2,3)],(3,2)], 'key_error' ),
                          ( [[(3,3),(2,1)],[(3,1),(3,2)]], 'broadcast_error' ) ) ),
     'dict': cartesian_prod( {'a':{'a':(2,2),'b':(1,2)},'b':{'a':(2,3),'b':(1,3)}},
                        (  ( {'a':(2,3),'b':(3,1)}, None ),
                           ( {'a':{'a':(2,1),'b':(2,3)},'b':(3,1)}, None ),
                           ( {'a':{'a':(2,1),'b':(2,3)},'b':{'a':(3,1),'b':(3,2)}}, None ),
                           ( {'a':(2,3),'b':(3,1),'c':(3,1)}, 'key_error' ) ) ),
     'mixed': cartesian_prod( [{'a':[(1,3),(2,3)],'b':[(1,3),(2,3)]},{'c':[(1,2),(2,2)],'d':[(1,2),(2,2)]}],
                        (  ( [(3,2),(2,1)], None ),
                           ( [{'a':(3,1),'b':(3,2)},{'a':(2,1),'b':(2,2)}], None ),
                           ( [{'a':[(3,1),(3,2)],'b':[(3,2),(3,1)]},{'a':[(2,1),(2,2)],'b':[(2,2),(2,1)]}], None ),
                           ( [(2,2),(2,1)], 'broadcast_error' ) ) ) 
    }   
    MATMULT_TEST_CASES = utils.group_first_n_params(MATMULT_TEST_CASES, n=2)
    @pytest.mark.parametrize('contents_list, exception', utils.unpack_test_cases(MATMULT_TEST_CASES), 
                                                         ids=utils.unpack_test_ids(MATMULT_TEST_CASES), 
                                                         indirect=['contents_list'])
    def test_matmult(self, contents_list, exception): 
        matmult_func = lambda x,y : x @ y
        self.perform_function_test(*contents_list, func=matmult_func, exception=exception)

    KRON_TEST_CASES = \
    {'list': cartesian_prod( [(2,3),[(2,1),(2,2)]], 
                           ( ( [(2,2), (2,1)], None ),
                             ( [(2,2),[(1,3),(2,1)]], None ),
                             ( [[(2,2)],[(2,2)]], 'key_error' ) ) ),
     'dict': cartesian_prod( {'a':(2,3), 'b':{'c':(2,1),'d':(2,2)}},
                           ( ( {'a':(2,2), 'b':(3,2)}, None ),
                             ( {'a':(2,2), 'b':{'c':(3,2),'d':(2,1)}}, None ),
                             ( {'a':(2,2), 'b':{'c':(3,2),'e':(2,1)}}, 'key_error' ) ) ),
     'mixed': cartesian_prod( [{'a':[(2,2),(1,3)],'b':{'c':(2,2)}}, {'a':[(2,1),(1,2)],'b':{'e':(2,2)}}],
                            ( ( [(2,1),(1,2)], None ),
                              ( [{'a':(1,2),'b':(2,1)},{'a':(1,3),'b':(3,1)}], None ),
                              ( [{'a':(1,2),'b':{'c':(2,1)}},{'a':(1,3),'b':(3,1)}], None ),
                              ( [{'a':(1,2),'b':{'e':(2,1)}},{'a':(1,3),'b':(3,1)}], 'key_error' ) ) )
    }
    KRON_TEST_CASES = utils.group_first_n_params(KRON_TEST_CASES, n=2)
    @pytest.mark.parametrize('contents_list, exception', utils.unpack_test_cases(KRON_TEST_CASES), 
                                                         ids=utils.unpack_test_ids(KRON_TEST_CASES), 
                                                         indirect=['contents_list'])
    def test_kron(self, contents_list, exception):
        kron_func = lambda x,y : np.kron(x, y)
        self.perform_function_test(*contents_list, func=kron_func, exception=exception)
    
    MEAN_AXIS_AND_EXCEPTIONS = ( (None, None), (0, None), (1, None), ((0,1), None), (2, Exception) )
    MEAN_TEST_CASES = \
    {'list': cartesian_prod( [(2,3,2),[(2,1),(2,2)]], MEAN_AXIS_AND_EXCEPTIONS),
     'dict': cartesian_prod( {'a':(2,3), 'b':{'c':(2,1,2),'d':(2,2)}}, MEAN_AXIS_AND_EXCEPTIONS ),
     'mixed': cartesian_prod( [{'a':[(2,2,2),(1,3)],'b':{'c':(2,2)}}, {'a':[(2,1,3),(1,2)],'b':{'e':(2,2)}}], MEAN_AXIS_AND_EXCEPTIONS )
    }
    @pytest.mark.parametrize('contents, axis, exception', utils.unpack_test_cases(MEAN_TEST_CASES), 
                                                          ids=utils.unpack_test_ids(MEAN_TEST_CASES), 
                                                          indirect=['contents'])
    def test_mean(self, contents, axis, exception):
        def mean_func(x, axis):
            result = np.mean(x, axis=axis)
            if isinstance(result, Number):
                result = self.array_constructor(result)
            return result

        self.perform_function_test(contents, func=mean_func, args=(axis,), exception=exception)
