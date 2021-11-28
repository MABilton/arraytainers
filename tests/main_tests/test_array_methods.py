import pytest
import numpy as np
from itertools import product

from .utils import apply_func_to_contents, assert_equal_values, assert_same_types, flatten_contents, sum_elements

ARRAY_CONVERSION = \
{'lone_numbers': [(1, [np.array(1)]), (-1.5, [np.array(-1.5)])],
'simple_lists': [([[2]], [np.array([[2]])])],
'simple_dicts': [({'a': 2}, {'a': np.array(2)})],
'list_with_array': [([[np.array(2)]], [[np.array(2)]])],
'list_with_dict_and_list': [([{'a': 2}, [[2,2],[2,2]]], [{'a': np.array(2)}, np.array([[2,2],[2,2]])])],
'dict_with_convertible_contents': [({'a':[[2]],'b':{'c':[3]}}, {'a':np.array([[2]]),'b':{'c':np.array([3])}}),
                                ({'a':[[np.array(2)]],'b':{'c':[3]}}, {'a':[[np.array(2)]],'b':{'c':np.array([3])}})]}

SUM_ARRAYS_CASES = \
{'empty': [*product(([], {}), (None,))],
'single_values': [*product(([(3,)], {'a': (3,)}, [[[(3,)]]], {'a':{'a':{'a':(3,)}}}, {'a':[{'a':(3,)}]}), (None,))],
'working_list': [*product((3*[(3,3)], [(2,2), 2*[(2,2)]], [3*[(2,2)], 2*[(2,2)]]), (None,))],
'list_broadcasting_error': [*product(([(3,3), (2,1)], [[(2,2), (3,1)], 2*[(2,2)]]), ('broadcast_error',))],
'working_dict': [*product(({'a': (1,3) , 'b': (3,3)}, {'a': (3,3), 'b':{'c': (3,3)}}), (None,))],
'dict_broadcasting_error': [*product(({'a': (3,2) , 'b': (3,3)}, {'a': (3,3), 'b':{'c': (3,2)}}), ('broadcast_error',))],
'working_mixed': [*product(([{'a':(1,3)}, [(3,3)]], {'a':{'c':[(3,3)]}, 'b':[(1,3), (3,3)]}), (None,))],
'mixed_broadcasting_error': [*product(([{'a':(1,2)}, [(3,3)]], {'a':{'c':[(3,1)]}, 'b':[(1,2), (3,3)]}), ('broadcast_error',))]}

SUM_ELEMENTS_CASES = \
{'empty': [*product(([], {}), (None,))],
'single_vals': [*product(([(3,)], {'a': (3,)}, [[[(3,)]]], {'a':{'a':{'a':(3,)}}}, {'a':[{'a':(3,)}]},
                        {'b':[(2,3), {'c':(3,3)}]}, [{'a':(3,2), 'b':[(1,2), (3,4)]}]), (None,))],
'list_working': [*product((3*[(3,3)], [(1,3), (3,3)], [[(3,3)], [(3,3)]], [(3,1), [(3,3), (3,3)]], 
                        [[(3,1)],[[[(3,1)]]]]), (None,))],
'list_key_error': [*product(([[(3,2)],[(3,3), (3,3)]], [[(3,3)],[[(3,3)], (3,3)]], 
                            [[(3,3), (3,3)],[(3,3), (3,3), (3,3)]]), ('key_error',))],
'list_broadcasting_error': [*product(([(3,3),(3,2)], [(3,2),[(3,3), (3,3)]]), ('broadcast_error',))],
'dict_working': [*product(({'a': (1,3) , 'b': (3,3)}, {'a': {'c':(3,3)}, 'b':{'c':(3,3)}}, 
                        {'a':(3,3),'b':{'c':(3,3)}}), (None,))],
'dict_broadcasting_error': [*product(({'a':(3,2),'b':(3,3)}, {'a':(3,2),'b':{'e':(3,3), 'd':(3,3)}}), ('broadcast_error',))],
'dict_key_error': [*product(({'a':{'c':(3,3)},'b':{'d':(3,3)}}, {'a':{'c':{'d':(3,3)}},'b':{'c':{'e':(3,3)}}}), ('key_error',))],
'mixed_working': [*product(([{'a':[(3,1)]}, {'a':[(3,1)]}], {'a':[(3,3), (3,3)], 'b':[(1,3), (3,3)]}, 
                            {'a':{'d':(3,3)},'b':{'d':(3,3)}}, {'b':[(1,3), {'c':(3,3)}]}, 
                            {'a':[(3,3), (3,3)], 'b':[[(3,3)], {'c':(3,3)}]}), (None,))],
'mixed_broadcasting_error': [*product(({'b':[(2,3), {'c':(3,3)}],'c':(3,3)}, 
                                    {'a':[(2,3), (3,3)], 'b':[[(3,3)], {'c':(3,3)}]}), ('broadcast_error',))],
'mixed_key_error': [*product(([{'a':[(3,1), (3,1)]}, {'a':[(3,1)]}], [{'a':[(3,1)]}, {'b':[(3,1)]}], 
                            {'a':[(3,3), (3,3)], 'b':[{'c':(3,3)}]}, {'a':[(3,3)],'b':{'c':(3,3)}}), ('key_error',))]}

class ArrayMixin:
    def test_shape_methods(self, std_contents):
        arraytainer = self.container_class(std_contents)

        shape_func = lambda array : array.shape
        expected, _ = apply_func_to_contents(std_contents, func=shape_func)

        assert_equal_values(arraytainer.shape, expected)
        assert_equal_values(arraytainer.shape_container.unpacked, expected)
        assert_same_types(arraytainer.shape_container, self.container_class(expected))

    @pytest.mark.parametrize('contents_in, expected', [val_i for val in ARRAY_CONVERSION.values() for val_i in val], 
                                                    ids=[key for key, val in ARRAY_CONVERSION.items() for _ in val])
    def test_array_conversion(self, contents_in, expected):
        arraytainer = self.container_class(contents_in)
        assert_equal_values(arraytainer.unpacked, expected)
        assert_same_types(arraytainer, self.container_class(expected))
    
    def test_array_methods_boolean(self, bool_contents):

        arraytainer = self.container_class(bool_contents)

        array_list = flatten_contents(bool_contents)

        assert arraytainer.all() == all([np.all(x) for x in array_list])
        assert arraytainer.any() == any([np.any(x) for x in array_list])

    @pytest.mark.parametrize('contents, exception', [val_i for val in SUM_ELEMENTS_CASES.values() for val_i in val], 
                                                    ids=[key for key, val in SUM_ELEMENTS_CASES.items() for _ in val],
                                                    indirect=['contents'])
    def test_array_methods_sum_elements(self, contents, exception):
        
        arraytainer = self.container_class(contents)

        if exception is None:
            expected = sum_elements(contents)
            # If expected is an integer value (i.e. if it's 0):
            expected = [expected] if not hasattr(expected, '__len__') else expected
            result = arraytainer.sum_elements()
            # For comparisons, need to 'unpack' sum_result if it's an arraytainer:
            assert_equal_values(result.unpacked, expected)
            assert_same_types(result, self.container_class(expected))
        else:
            self.assert_exception(lambda x: x.sum_elements(), exception, arraytainer)

    @pytest.mark.parametrize('contents,exception', [val_i for val in SUM_ARRAYS_CASES.values() for val_i in val], 
                                                    ids=[key for key, val in SUM_ARRAYS_CASES.items() for _ in val],
                                                    indirect=['contents'])
    def test_array_methods_sum_arrays(self, contents, exception):

        arraytainer = self.container_class(contents)
        
        if exception is None:
            array_list = flatten_contents(contents)
            expected = sum(array_list)
            result = arraytainer.sum_arrays()
            assert np.allclose(result, expected)
            assert any([isinstance(result, type_i) for type_i in self.array_types])
        else:
            self.assert_exception(lambda x: x.sum_arrays(), exception, arraytainer)