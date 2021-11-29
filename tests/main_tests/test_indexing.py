import pytest
import numpy as np

from .utils import apply_func_to_contents, assert_equal_values, assert_same_types, create_idx_combos

INDEX_HASH = \
{'simple_dict': create_idx_combos({'a':(2,3),'b':(2,1)}, [('a',None),(0,'key_error'),(-1,'key_error'),('c','key_error')]),
    'simple_list': create_idx_combos([(2,3),(2,1)], [('a','key_error'),(0,None),(-1,None),(3,'key_error')]),
    'nested_list': create_idx_combos([[(2,3),(2,3),(2,3)],(2,1)], [(0,None),(1,None),(2,'key_error')]),
    'nested_dict': create_idx_combos({'a':(2,1),'b':{'c':(2,2),'d':(2,1)},'c':(1,1)}, 
                                    [('a',None),('b',None),('c',None),('d','key_error')]),
    'mixed': create_idx_combos({'a':[(1,2),(2,3),{'c':(3,2)}], 'b':{'c':[(2,1), (2,2)]}}, 
                            [('a',None),('b',None),(0,'key_error'),('c','key_error')])}

INDEX_SLICE = \
{'simple_dict': create_idx_combos({'a':(6,),'b':(7,)}, [(slice(1),None), (slice(1,5),None),(slice(1,10,2),None),
                                    ((slice(1,5,2), slice(1,5,2)),'key_error')]),
    'simple_list': create_idx_combos([(3,3),(2,2)], [(slice(1),None), (slice(0,2),None), ((slice(0),slice(0,2)),None)]),
    'nested_list': create_idx_combos([[(2,2,2), (2,)],(2,1)], [(slice(0,2), None), ((slice(0,0),slice(0,0)),'key_error')]),
    'nested_dict': create_idx_combos({'a':(2,1),'b':{'c':(2,2),'d':(2,)},'c':(1,1)}, 
                                    [(slice(0,2), None), ((slice(0,0),slice(0,0)),'key_error')]),
    'mixed': create_idx_combos({'a':[(1,2),(3,),{'c':(3,2)}], 'b':{'c':[(2,1), (2,2)]}}, 
                            [(slice(0,2), None), ((slice(0,0),slice(0,0)),'key_error')])}

INDEX_ARRAY = \
{'simple_dict': create_idx_combos({'a':(3,),'b':(3,)}, [([0,-1],None), ([],'key_error'),
                                    ([5],'key_error'), (np.array([True, False, True]),None), 
                                    ([True, False],'key_error'), ([[True, False], [False, True]],'key_error')]),
    'simple_list': create_idx_combos([(3,3),(2,2)], [([0,-1],None), ([[0,-1],[0,-1]],None),
                                        ([[10,0],[0,-2]],'key_error'), ([[True,False],[True,False]],'key_error')])} #,
#  'nested_list': create_array_combos([[(2,2,2), (2,)],(2,1)], [(np.array([0,1]), None), 
#                                     ((np.array([0,1]),np.array([0,1])),'key_error')]),
#  'nested_dict': create_array_combos({'a':(2,1),'b':{'c':(2,2),'d':(2,)},'c':(1,1)}, 
#                                     [(np.array([0,1]), None), ((np.array([0,1]),np.array([0,1])),'key_error')]),
#  'mixed': create_array_combos({'a':[(1,2),(3,),{'c':(3,2)}], 'b':{'c':[(2,1), (2,2)]}}, 
#                               [(np.array([0,1]), None), ((np.array([0,1]),np.array([0,1])),'key_error')])}

class IndexMixin:
    
    @pytest.mark.parametrize('contents, key, exception', [val_i for val in INDEX_HASH.values() for val_i in val], 
                                                         ids=[key for key, val in INDEX_HASH.items() for _ in val],
                                                         indirect=['contents'])
    def test_indexing_with_hash(self, contents, key, exception):
        
        arraytainer = self.container_class(contents)

        if exception is None:
            result = arraytainer[key]
            expected = contents[key]
            if isinstance(result, self.container_class):
                assert_equal_values(result.unpacked, expected)
                assert_same_types(result, self.container_class(expected))
            else:
                assert_equal_values(result, expected)
        else:
            self.assert_exception(lambda x, key: x[key], exception, arraytainer, key)

    @pytest.mark.parametrize('contents, slice_val, exception', [val_i for val in INDEX_SLICE.values() for val_i in val], 
                                                               ids=[key for key, val in INDEX_SLICE.items() for _ in val],
                                                               indirect=['contents'])
    def test_indexing_with_slice(self, contents, slice_val, exception):
        
        arraytainer = self.container_class(contents)
        index_func = lambda contents : contents[slice_val]

        if exception is None:
            result = arraytainer[slice_val]
            expected, _ = apply_func_to_contents(contents, func=index_func)
            assert_equal_values(result.unpacked, expected)
            assert_same_types(result, self.container_class(expected))
        else:
            self.assert_exception(lambda x, key: x[key], exception, arraytainer, slice_val)

    # @pytest.mark.parametrize('contents, array_vals, exception', 
    #                          [val_i for val in INDEX_ARRAY.values() for val_i in val], 
    #                          ids=[key for key, val in INDEX_ARRAY.items() for _ in val],
    #                          indirect=['contents'])
    # def test_indexing_with_array(self, contents, array_vals, exception):

    #     arraytainer = self.container_class(contents)
    #     array_key = self.array(array_vals)

    #     # There are errors when trying to index a Numpy array with a Jax array
    #     # and vice versa - need to make sure that Arraytainers can handle these
    #     # cases properly:
    #     def index_func(in_contents, array_key):
    #         try:
    #             indexed = in_contents[array_key]
    #         except:
    #             indexed = in_contents[self.array(array_key)]
    #         return indexed 

    #     # # Strangely, indexing Jax arrays with Jax arrays which have out-of-bounds indices does
    #     # # NOT throw an exception - this is meant to be 'normal behaviour:
    #     # if isinstance(array_class(1), jnp.DeviceArray) and ('jax' in self.container_class.__name__.lower()):
    #     #     # In case of indexing Jax array with another Jax array, we'll work out what the error is
    #     #     # supposed to be:
    #     #     expected, exception = apply_func_to_contents(contents, func=index_func, args=(array_key,))

    #     if exception is None:
    #         result = arraytainer[array_key]
    #         expected, _ = apply_func_to_contents(contents, func=index_func, args=(array_key,))
    #         assert_equal_values(result.unpacked, expected)
    #         assert_same_types(result, self.container_class(expected))
    #     else:
    #         self.assert_exception(lambda x, key: x[key], exception, arraytainer, array_key)



    # def test_indexing_with_arraytainer(self, arraytainers_contents_keys):
        
    #     arraytainer = self.container_class(contents)
    #     arraytainer_key = self.container_class(key_contents)
        
    #     index_func = lambda contents, idx : contents[idx][key_contents[idx]]
    #     expected, exception = apply_func_to_contents(contents, func=index_func)
        
    #     with exception:
    #         assert arraytainer[arraytainer_key] == self.container_class(expected)