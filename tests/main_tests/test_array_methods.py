import pytest
import numpy as np
from copy import deepcopy
from itertools import product
import operator

from . import utils
from .utils import cartesian_prod

class ArrayMixin:

    def test_arraytainer_composition(self, std_contents):
        arraytainer = self.container_class(std_contents)
        arraytainer_2 = self.container_class(arraytainer)
        utils.assert_equal_values(arraytainer.unpacked, arraytainer_2.unpacked)
        utils.assert_same_types(arraytainer, arraytainer_2)

    def test_convert_array_types(self, std_shapes):
        for array_constructor in utils.ARRAY_CONSTRUCTORS:
            contents = utils.create_contents(std_shapes, array_constructor)
            arraytainer = self.container_class(contents)
            arrays_in_arraytainer = utils.get_list_of_arrays(arraytainer.unpacked)
            assert all([isinstance(x, self.expected_array_types) for x in arrays_in_arraytainer])

    def test_shape_methods(self, std_contents_and_shapes):

        in_contents, shapes = std_contents_and_shapes

        arraytainer = self.container_class(in_contents)

        expected = self.container_class(shapes, greedy_array_conversion=True)
        result = arraytainer.shape

        utils.assert_equal_values(result.unpacked, expected.unpacked)
        utils.assert_same_types(expected, result)

    def test_transpose(self, std_contents):
        arraytainer = self.container_class(std_contents)
        transpose_func = lambda x : x.T

        expected, exception = utils.apply_func_to_contents(std_contents, func=transpose_func)
        if exception is None:
            result = transpose_func(arraytainer)
            utils.assert_equal_values(result.unpacked, expected, approx_equal=True)
            utils.assert_same_types(result, self.container_class(expected))
        else:
            self.assert_exception(transpose_func, exception, arraytainer)

    def test_boolean_methods(self, bool_contents):

        arraytainer = self.container_class(bool_contents)

        array_list = utils.get_list_of_arrays(bool_contents)

        assert arraytainer.all() == all([np.all(x) for x in array_list])
        assert arraytainer.any() == any([np.any(x) for x in array_list])

    INCREMENT_FUNCS = {'addition': lambda x,y: operator.iadd(x,y), # Equivalent to x += y
                       'subtraction': lambda x,y: operator.isub(x,y), # Equivalent to x -= y
                       'multiplication': lambda x,y: operator.imul(x,y), # Equivalent to x *= y
                       'division': lambda x,y: operator.itruediv(x,y), # Equivalent to x /= y
                       'power': lambda x,y: operator.ipow(x,y)} # Equivalent to x **= y
    @pytest.mark.parametrize('func', INCREMENT_FUNCS.values(), ids=INCREMENT_FUNCS.keys())
    @pytest.mark.parametrize('val', [2, 2.0, np.array([2.0])], ids=['int', 'float', 'array'])
    def test_increment_with_scalar_or_array(self, std_contents, val, func):
        arraytainer = self.container_class(std_contents)
        key = self.array_constructor(val) if utils.is_array(val) else val
        expected, _ = utils.apply_func_to_contents(std_contents, func=func, args=(key,), throw_exception=True)
        result = func(arraytainer, key)
        utils.assert_equal_values(expected, result.unpacked)
        utils.assert_same_types(self.container_class(expected), result)

    RESHAPE_TEST_CASES = {
    'reshape_list_using_tuple': cartesian_prod( [(6,2),(1,12)], ( ((3,4),None), ((12,), None), ((2,4),'reshape_error') ), 'tuple' ),
    'reshape_dict_using_tuple': cartesian_prod( {'a':(2,2),'b':(4,1)}, ( ((1,4),None), ((2,2), None), ((2,4),'reshape_error') ), 'tuple' ),
    'reshape_mixed_1_using_tuple': cartesian_prod( {'a':[{'a':[(12,),(1,12)]}],'b':[{'a':[(3,4),(4,3)]}]}, 
                                                                ( ((4,3),None), ((2,6),None), ((2,4),'reshape_error') ), 'tuple' ),
    'reshape_mixed_2_using_tuple': cartesian_prod( [{'a':(4,3),'b':[(3,4),(12,1)]},{'a':[(12,1),(1,12)]}] ,
                                                                ( ((4,3),None), ((2,6),None), ((2,4),'reshape_error') ), 'tuple' ),
    'reshape_list_using_arraytainer': cartesian_prod( [(3,4),(1,4)], 
                                                        ( ([(4,3),(2,2)],None), 
                                                        ([(12,),(2,2)],None), 
                                                        ({'a':(4,3),'b':(2,2)},'key_error'), 
                                                        ([(4,3),(2,2),(2,2)],'key_error'),
                                                        ([(12,1),(1,13)],'reshape_error') ), 'arraytainer' ),
    'reshape_dict_using_arraytainer': cartesian_prod( {'a':(2,2),'b':(4,1)}, 
                                                      ( ({'a':(4,1),'b':(2,2)},None), 
                                                        ([(4,1),(2,2)],'key_error'),
                                                        ({'a':(4,1),'b':(2,2),'c':(2,2)},'key_error'),
                                                        ({'a':{'c':(4,1)},'b':(2,2)},'key_error'),
                                                        ({'a':[(4,1)],'b':(2,2)},'key_error'),
                                                        ({'a':(5,1),'b':(2,2)},'reshape_error') ), 'arraytainer' ),
    'reshape_mixed_1_using_arraytainer': cartesian_prod( {'a':[{'a':[(12,),(1,12)]}],'b':[{'a':[(3,4),(4,3)]}]}, 
                                            ( ({'a':[{'a':[(6,2),(4,3)]}],'b':[{'a':[(12,1),(1,12)]}]},None),
                                                ({'a':[{'a':[(5,2),(4,3)]}],'b':[{'a':[(12,1),(1,12)]}]},'reshape_error'),
                                                ({'a':{'a':[(6,2),(4,3)]},'b':[{'a':[(12,1),(1,12)]}]},'key_error') ), 'arraytainer' ),
    'reshape_mixed_2_using_arraytainer': cartesian_prod( [{'a':(4,3),'b':[(3,4),(12,1)]},{'a':[(12,1),(1,12)]}] ,
                                              ( ([{'a':(2,6),'b':[(12,),(6,2)]},{'a':[(4,3),(3,4)]}],None),
                                                ([{'a':(3,6),'b':[(12,),(6,2)]},{'a':[(4,3),(3,4)]}],'reshape_error'),
                                                ([{'a':[(3,6)],'b':[(12,),(6,2)]},{'a':[(4,3),(3,4)]}],'key_error') ), 'arraytainer' ),
    'reshape_2_key_broadcasting': cartesian_prod( {'a':[{'c':(4,1)},[(2,2)]],'b':[(3,3),(9,1)]}, 
                                                ( ({'a':(1,4),'b':(1,9)},None), 
                                                  ({'a':[(1,4),(4,1)],'b':[(1,9),(3,3)]},None),
                                                  ({'a':(1,5),'b':(1,9)},'reshape_error'), 
                                                  ({'a':[(1,4)],'b':[(1,9),(3,3)]},'key_error') ), 'arraytainer'),
    'reshape_2_key_broadcasting': cartesian_prod( [{'a':[(2,4),(8,)],'b':(8,1)},{'c':{'d':(2,3),'e':(3,2)}}], 
                                                  ( ([(1,8),(6,1)],None), 
                                                    ([{'a':(4,2),'b':(1,8)},{'c':(6,)}],None),
                                                    ([{'a':[(1,8),(2,4)],'b':(1,8)}, {'c':(6,)}],None),
                                                    ([{'a':[(2,4)],'b':(1,8)}, {'c':(6,)}], 'key_error'),
                                                    ({'a':[(1,8),(2,4)],'b':(1,8)}, 'key_error') ), 'arraytainer')
    }
    @pytest.mark.parametrize('contents, new_shape, exception, new_shape_type', 
                             utils.unpack_test_cases(RESHAPE_TEST_CASES), 
                             ids=utils.unpack_test_ids(RESHAPE_TEST_CASES),
                             indirect=['contents'])
    def test_reshape(self, contents, new_shape, exception, new_shape_type):
        
        new_shape = deepcopy(new_shape)
        arraytainer = self.container_class(contents)
        
        reshape_func = lambda x, new_shape: x.reshape(new_shape)

        if exception is None:
            index_args_fun = lambda args, idx : args if isinstance(args[0], tuple) else (args[0][idx],)
            expected, _ = utils.apply_func_to_contents(contents, args=(new_shape,), func=reshape_func, 
                                                       index_args_fun=index_args_fun, throw_exception=True)
            # Convert new_shape to container AFTER computing expected result, since utils.apply_func_to_contents
            # does not accept arraytainers as inputs:
            new_shape = self.container_class(new_shape, greedy_array_conversion=True) if new_shape_type=='arraytainer' else new_shape
            result = arraytainer.reshape(new_shape)
            utils.assert_equal_values(result.unpacked, expected)
            utils.assert_same_types(result, self.container_class(expected))
            
            # If using a tuple to reshape, 
            if new_shape_type=='tuple':
                arraytainer_2 = self.container_class(contents)
                result_2 = arraytainer.reshape(*new_shape)
                utils.assert_equal_values(result_2.unpacked, expected)
                utils.assert_same_types(result_2, self.container_class(expected))

        else:
            new_shape = self.container_class(new_shape, greedy_array_conversion=True) if new_shape_type=='arraytainer' else new_shape
            self.assert_exception(reshape_func, exception, arraytainer, new_shape)

    ARRAY_CONVERSION_TEST_CASES = {
    'lone_numbers': [ (1, [np.array(1)]), 
                      (-1.5, [np.array(-1.5)]) ],
    'simple_lists': [ ([[2]], [np.array([[2]])]), 
                      ([[1,2]], [np.array([1,2])]), 
                      ([[1,2],[3,4]], [np.array([1,2]),np.array([3,4])]),
                      ([[[1,2],[3,4]]], [np.array([[1,2],[3,4]])]) ],
    'simple_dicts': [ ({'a': 2}, {'a': np.array(2)}), 
                      ({'a':2,'b': [[3]]}, {'a':np.array(2),'b':np.array([[3]])}) ],
    'list_with_array': [ ([[np.array(2)]], [[np.array(2)]]) ],
    'list_with_dict_and_list': [ ([{'a': 2}, [[2,2],[2,2]]], [{'a': np.array(2)}, np.array([[2,2],[2,2]])]) ],
    'dict_with_convertible_contents': [ ({'a':[[2]],'b':{'c':[3]}}, {'a':np.array([[2]]),'b':{'c':np.array([3])}}),
                                        ({'a':[[np.array(2)]],'b':{'c':[3]}}, {'a':[[np.array(2)]],'b':{'c':np.array([3])}}) ]
    }
    @pytest.mark.parametrize('contents_in, expected', utils.unpack_test_cases(ARRAY_CONVERSION_TEST_CASES), 
                                                      ids=utils.unpack_test_ids(ARRAY_CONVERSION_TEST_CASES))
    def test_convert_list_to_array(self, contents_in, expected):
        arraytainer = self.container_class(contents_in)
        utils.assert_equal_values(arraytainer.unpacked, expected)
        utils.assert_same_types(arraytainer, self.container_class(expected))

    SUM_ELEMENTS_TEST_CASES = {
    'empty': cartesian_prod( ( [], {} ), None ),
    'single_values': cartesian_prod( ( [(3,)], {'a':(3,)}, [[[(3,)]]], 
                                    {'a':{'a':{'a':(3,)}}}, {'a':[{'a':(3,)}]},
                                    {'b':[(2,3),{'c':(3,3)}]}, [{'a':(3,2),'b':[(1,2),(3,4)]}] ), None),
    'list_working': cartesian_prod( ( [(3,3),(3,3)], 
                                    [(1,3),(3,3)], 
                                    [[(3,3)],[(3,3)]], 
                                    [(3,1),[(3,3),(3,3)]], 
                                    [[(3,1)],[[[(3,1)]]]] ), None ),
    'list_key_error': cartesian_prod( ( [[(3,2)],[(3,3), (3,3)]], 
                                        [[(3,3)],[[(3,3)], (3,3)]], 
                                        [[(3,3), (3,3)],[(3,3), (3,3), (3,3)]] ), 'key_error' ),
    'list_broadcasting_error': cartesian_prod( ( [(3,3),(3,2)], 
                                                [(3,2),[(3,3),(3,3)]] ), 'broadcast_error' ),
    'dict_working': cartesian_prod( ( {'a':(1,3),'b':(3,3)}, 
                                    {'a':{'c':(3,3)},'b':{'c':(3,3)}}, 
                                    {'a':(3,3),'b':{'c':(3,3)}} ), None ),
    'dict_broadcasting_error': cartesian_prod( ( {'a':(3,2),'b':(3,3)}, 
                                                {'a':(3,2),'b':{'e':(3,3), 'd':(3,3)}} ), 'broadcast_error' ),
    'dict_key_error': cartesian_prod( ( {'a':{'c':(3,3)},'b':{'d':(3,3)}}, 
                                        {'a':{'c':{'d':(3,3)}},'b':{'c':{'e':(3,3)}}} ), 'key_error' ),
    'mixed_working': cartesian_prod( ( [{'a':[(3,1)]}, {'a':[(3,1)]}], 
                                    {'a':[(3,3),(3,3)], 'b':[(1,3),(3,3)]}, 
                                    {'a':{'d':(3,3)},'b':{'d':(3,3)}},
                                    {'b':[(1,3),{'c':(3,3)}]}, 
                                    {'a':[(3,3),(3,3)],'b':[[(3,3)],{'c':(3,3)}]} ), None ),
    'mixed_broadcasting_error': cartesian_prod( ( {'b':[(2,3),{'c':(3,3)}],'c':(3,3)}, 
                                                {'a':[(2,3),(3,3)],'b':[[(3,3)],{'c':(3,3)}]} ), 'broadcast_error' ),
    'mixed_key_error': cartesian_prod( ( [{'a':[(3,1), (3,1)]}, {'a':[(3,1)]}], 
                                        [{'a':[(3,1)]}, {'b':[(3,1)]}], 
                                        {'a':[(3,3),(3,3)],'b':[{'c':(3,3)}]}, 
                                        {'a':[(3,3)],'b':{'c':(3,3)}} ), 'key_error' )
    }
    @pytest.mark.parametrize('contents, exception', utils.unpack_test_cases(SUM_ELEMENTS_TEST_CASES), 
                                                    ids=utils.unpack_test_ids(SUM_ELEMENTS_TEST_CASES),
                                                    indirect=['contents'])
    def test_sum_elements_method(self, contents, exception):
        
        arraytainer = self.container_class(contents)

        if exception is None:
            expected = utils.sum_elements(contents)
            # If expected is not a list or array, we need to 'package' it as a list, since
            # arraytainers will automatically place 'lone' entries in a list:
            expected = [expected] if not isinstance(expected,(list,dict)) else expected
            result = arraytainer.sum_elements()
            # For comparisons, need to 'unpack' sum_result if it's an arraytainer:
            try:
                utils.assert_equal_values(result.unpacked, expected)
                utils.assert_same_types(result, self.container_class(expected))
            except:
                raise Exception
        else:
            self.assert_exception(lambda x: x.sum_elements(), exception, arraytainer)

    SUM_ARRAYS_TEST_CASES = {
    'empty': cartesian_prod( ( [], {} ) , None ),
    'single_values': cartesian_prod( ( [(3,)], {'a': (3,)}, [[[(3,)]]], 
                                       {'a':{'a':{'a':(3,)}}}, {'a':[{'a':(3,)}]} ), None ),
    'working_list': cartesian_prod( ( [(3,3),(3,3),(3,3)], 
                                      [(2,2), [(2,2), (2,2)]], 
                                      [[(2,2),(2,2)],[(2,2),(2,2)]] ), None ),
    'list_broadcasting_error': cartesian_prod( ( [(3,3),(2,1)], 
                                                 [[(2,2),(3,1)],[(2,2),(2,2)]] ), 'broadcast_error' ),
    'working_dict': cartesian_prod( ( {'a':(1,3),'b':(3,3)}, 
                                      {'a':(3,3),'b':{'c':(3,3)}} ), None),
    'dict_broadcasting_error': cartesian_prod( ( {'a':(3,2),'b':(3,3)}, 
                                                 {'a':(3,3),'b':{'c':(3,2)}} ), 'broadcast_error'),
    'working_mixed': cartesian_prod( ( [{'a':(1,3)}, [(3,3)]], 
                                       {'a':{'c':[(3,3)]}, 'b':[(1,3), (3,3)]} ), None ),
    'mixed_broadcasting_error': cartesian_prod( ( [{'a':(1,2)},[(3,3)]], 
                                                  {'a':{'c':[(3,1)]},'b':[(1,2),(3,3)]} ),  'broadcast_error')
    }
    @pytest.mark.parametrize('contents, exception', utils.unpack_test_cases(SUM_ARRAYS_TEST_CASES), 
                                                    ids=utils.unpack_test_ids(SUM_ARRAYS_TEST_CASES),
                                                    indirect=['contents'])
    def test_sum_arrays_method(self, contents, exception):

        arraytainer = self.container_class(contents)
        
        if exception is None:
            array_list = utils.get_list_of_arrays(contents)
            expected = sum(array_list)
            result = arraytainer.sum_arrays()
            assert np.allclose(result, expected)
            assert utils.is_array(result)
        else:
            self.assert_exception(lambda x: x.sum_arrays(), exception, arraytainer)