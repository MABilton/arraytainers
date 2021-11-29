import pytest
import numpy as np
from itertools import product

from .utils import apply_func_to_contents, assert_equal_values, assert_same_types, get_keys

ARRAY_CONVERSION = {
'lone_numbers': [ 
                   (1, [np.array(1)]), 
                   (-1.5, [np.array(-1.5)]) 
                ],
'simple_lists': [ 
                  ([[2]], [np.array([[2]])]), 
                  ([[1,2]], [np.array([1,2])]), 
                  ([[1,2],[3,4]], [np.array([1,2]),np.array([3,4])]),
                  ([[[1,2],[3,4]]], [np.array([[1,2],[3,4]])]) 
                ],
'simple_dicts': [
                 ({'a': 2}, {'a': np.array(2)}), 
                 ({'a':2,'b': [[3]]}, {'a':np.array(2),'b':np.array([[3]])})
                ],
'list_with_array': [ ([[np.array(2)]], [[np.array(2)]]) ],
'list_with_dict_and_list': [ ([{'a': 2}, [[2,2],[2,2]]], [{'a': np.array(2)}, np.array([[2,2],[2,2]])]) ],
'dict_with_convertible_contents': [ 
                                   ({'a':[[2]],'b':{'c':[3]}}, {'a':np.array([[2]]),'b':{'c':np.array([3])}}),
                                   ({'a':[[np.array(2)]],'b':{'c':[3]}}, {'a':[[np.array(2)]],'b':{'c':np.array([3])}}) 
                                  ]
}

# RESHAPE_CASES = {
#     'using_tuple': [ ( [(6,2), (1,12) ], (3,4), None ),
#                      ({'a': (2,3), 'b': (3,2)}, (3,2), None),
#                      ({'a': [{'x': (12,)}], 'b': [{'a': (3,4)}]}, (2,6), None),
#                      ( [(6,2), (1,12) ], (3,4), 'reshape_error'),
#                      ({'a': [{'x': (13,)}], 'b': [{'a': (3,4)}]}, (2,6), 'reshape_error')
#                     ],
#     'using_container': [ ([(6,2), (1,12) ], [(3,4), (12,1)], None),
#                          ({'a': (2,3), 'b': (3,2)}, {'a':(3,2),'b':(2,3)}, None),
#                          ({'a': [{'x': (12,)}], 'b': [{'a': (3,2)}]}, {'a': [{'x': (2,6)}], 'b': [{'a': (6,1)}]}, None),
#                          ({'a': [{'x': (12,), 'y': (2,6)}], 'b':[{'a': (3,2)}]} {'a':(3,4), 'b':(12,1)}, None)
#     'container_broadcasting':
#     ]
# }

SUM_ARRAYS_CASES = {
'empty': [ *product(([], {}), (None,)) ],
'single_values': [ *product(([(3,)], {'a': (3,)}, [[[(3,)]]], {'a':{'a':{'a':(3,)}}}, {'a':[{'a':(3,)}]}), (None,)) ],
'working_list': [ *product((3*[(3,3)], [(2,2), 2*[(2,2)]], [3*[(2,2)], 2*[(2,2)]]), (None,)) ],
'list_broadcasting_error': [ *product(([(3,3), (2,1)], [[(2,2), (3,1)], 2*[(2,2)]]), ('broadcast_error',)) ],
'working_dict': [ *product(({'a': (1,3) , 'b': (3,3)}, {'a': (3,3), 'b':{'c': (3,3)}}), (None,)) ],
'dict_broadcasting_error': [ *product(({'a': (3,2) , 'b': (3,3)}, {'a': (3,3), 'b':{'c': (3,2)}}), ('broadcast_error',)) ],
'working_mixed': [ *product(([{'a':(1,3)}, [(3,3)]], {'a':{'c':[(3,3)]}, 'b':[(1,3), (3,3)]}), (None,)) ],
'mixed_broadcasting_error': [ *product(([{'a':(1,2)}, [(3,3)]], {'a':{'c':[(3,1)]}, 'b':[(1,2), (3,3)]}), ('broadcast_error',)) ]
}

SUM_ELEMENTS_CASES = {
'empty': [*product(([], {}), (None,))],
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
                            {'a':[(3,3), (3,3)], 'b':[{'c':(3,3)}]}, {'a':[(3,3)],'b':{'c':(3,3)}}), ('key_error',))]
}

class ArrayMixin:
    def test_shape_methods(self, std_contents_and_shapes):

        in_contents, shapes = std_contents_and_shapes
        
        # in_contents, shapes = deepcopy(in_contents), deepcopy(shapes)

        arraytainer = self.container_class(in_contents)

        expected = self.container_class(shapes, greedy_array_conversion=True)
        result = arraytainer.shape

        assert_equal_values(result.unpacked, expected.unpacked)
        assert_same_types(expected, result)

    def test_array_methods_boolean(self, bool_contents):

        arraytainer = self.container_class(bool_contents)

        array_list = get_list_of_arrays(bool_contents)

        assert arraytainer.all() == all([np.all(x) for x in array_list])
        assert arraytainer.any() == any([np.any(x) for x in array_list])

    # @pytest.mark.parametrize('contents, new_shape, new_shape_is_container, exception', 
    #                          [val_i for val in ARRAY_CONVERSION.values() for val_i in val], 
    #                          ids=[key for key, val in ARRAY_CONVERSION.items() for _ in val])
    # def test_reshape_and_flatten(contents, new_shape, exception):
    #     arraytainer = self.container_class(contents)

    #     if not isinstance(new_shape, tuple):
    #         new_shape = self.container_class(new_shape)

    @pytest.mark.parametrize('contents_in, expected', [val_i for val in ARRAY_CONVERSION.values() for val_i in val], 
                                                    ids=[key for key, val in ARRAY_CONVERSION.items() for _ in val])
    def test_array_conversion(self, contents_in, expected):
        arraytainer = self.container_class(contents_in)
        assert_equal_values(arraytainer.unpacked, expected)
        assert_same_types(arraytainer, self.container_class(expected))

    # @pytest.mark.parametrize('contents, exception', [val_i for val in SUM_ELEMENTS_CASES.values() for val_i in val], 
    #                                                 ids=[key for key, val in SUM_ELEMENTS_CASES.items() for _ in val],
    #                                                 indirect=['contents'])
    # def test_array_methods_sum_elements(self, contents, exception):
        
    #     arraytainer = self.container_class(contents)

    #     if exception is None:
    #         expected = sum_elements(contents)
    #         # If expected is an integer value (i.e. if it's 0):
    #         expected = [expected] if not hasattr(expected, '__len__') else expected
    #         result = arraytainer.sum_elements()
    #         # For comparisons, need to 'unpack' sum_result if it's an arraytainer:
    #         assert_equal_values(result.unpacked, expected)
    #         assert_same_types(result, self.container_class(expected))
    #     else:
    #         self.assert_exception(lambda x: x.sum_elements(), exception, arraytainer)

    @pytest.mark.parametrize('contents,exception', [val_i for val in SUM_ARRAYS_CASES.values() for val_i in val], 
                                                    ids=[key for key, val in SUM_ARRAYS_CASES.items() for _ in val],
                                                    indirect=['contents'])
    def test_array_methods_sum_arrays(self, contents, exception):

        arraytainer = self.container_class(contents)
        
        if exception is None:
            array_list = get_list_of_arrays(contents)
            expected = sum(array_list)
            result = arraytainer.sum_arrays()
            assert np.allclose(result, expected)
            assert any([isinstance(result, type_i) for type_i in self.array_types])
        else:
            self.assert_exception(lambda x: x.sum_arrays(), exception, arraytainer)

# Helper functions:

# Adds up all of the elements in an (unpacked):
def sum_elements(contents):

    keys = get_keys(contents)
    content_list = [contents[key] for key in keys]
    
    array_elems = {key:val for key, val in zip(keys,content_list) if isinstance(val, ARRAY_TYPES)}
    nonarray_elems = [val for key,val in zip(keys,content_list) if key not in array_elems.keys()]
    array_elems = array_elems.values()
    
    if not nonarray_elems:
        sum_result = sum(content_list)
    else:
        elem_keys = get_keys(nonarray_elems[0])
        sum_result = {key: sum_elements([*[val[key] for val in nonarray_elems], 
                                         *[val for val in array_elems]]) 
                     for key in elem_keys}
        if isinstance(nonarray_elems[0], list):
            sum_result = list(sum_result.values())
    return sum_result

def get_list_of_arrays(contents, array_list=None):

    array_list = [] if array_list is None else array_list
    keys = get_keys(contents)

    for key in keys:
        if isinstance(contents[key], (dict, list)):
            array_list = get_list_of_arrays(contents[key], array_list)
        else:
            array_list.append(contents[key])
    
    return array_list