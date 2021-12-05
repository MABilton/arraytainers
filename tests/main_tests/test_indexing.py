import pytest
import numpy as np
from . import utils 
from .utils import cartesian_prod

class IndexMixin:
    GET_TEST_CASES = \
    {'list': cartesian_prod( [[(2,1),[(2,),(1,)]],[(2,2),[(2,),(1,)]]], 
                                ( ((0,), None), ((0,1), None) , ((0,1,0), None), ((1,1,1), None),
                                  ((2,), 'key_error'), ((0,0,0), AttributeError), ((1,1,2), 'key_error'), ((),'key_error') ) ),
     'dict': cartesian_prod( {'a':{'c':{1:(1,2)},0:(2,2)},'b':{'a':(3,2),'b':{('a','b'):(2,1)}}},
                                ( (('a',), None), (('a','c',1), None), (('b','b'), None),
                                  (('a',0), None), (('a','c','b'), 'key_error'), ((0,'c','b'), 'key_error'),
                                  (('b','b',('a','b')), None), (('b','a','b'), AttributeError) , ((),'key_error') ) ),
     'mixed': cartesian_prod( {'a':[(1,2),{1:(2,2),('a','b'):(2,3)},(2,2)],1:{'c':[(2,1),(1,2)],'d':(2,1)}},
                                 ( (('a',), None), (('a',1),None), (('a',1,1),None), (('a',1,('a','b')),None), ((1,),None), 
                                   ((1,'c'),None), ((1,'c',0),None), (('a',1,2),'key_error'), ((1,'c',2),'key_error'), 
                                   ((),'key_error') ) ) 
    }
    @pytest.mark.parametrize('contents, key_iterable, exception', utils.unpack_test_cases(GET_TEST_CASES), 
                                                                  ids=utils.unpack_test_ids(GET_TEST_CASES),
                                                                  indirect=['contents'])
    def test_get_method(self, contents, key_iterable, exception):
        arraytainer = self.container_class(contents)

        if exception is None:
            result = arraytainer.get(*key_iterable)
            expected = utils.get_contents_item(contents, key_iterable)
            result_val_check = result.unpacked if utils.is_arraytainer(result) else result
            utils.assert_equal_values(result_val_check, expected)
            result_type_check = result if utils.is_arraytainer(result) else self.container_class(result)
            utils.assert_same_types(result_type_check, self.container_class(expected))
        else:
            self.assert_exception(lambda x, key_iterable: x.get(*key_iterable), exception, arraytainer, key_iterable)

    HASH_INDEXING_TEST_CASES = \
    {'simple_dict': cartesian_prod( {'a':(2,3),'b':(2,1)}, ( ('a',None), (0,'key_error'), (-1,'key_error'), ('c','key_error') ) ),
     'simple_list': cartesian_prod( [(2,3),(2,1)], ( ('a','key_error'), (0,None), (-1,None), (2,'key_error') ) ),
     'nested_list': cartesian_prod( [[(2,3),(2,3),(2,3)],(2,1)], ( (0,None), (1,None), (2,'key_error') ) ),
     'nested_dict': cartesian_prod( {'a':(2,1),'b':{'c':(2,2),'d':(2,1)},'c':(1,1)} , 
                                        ( ('a',None), ('b',None), ('c',None), ('d','key_error') )),
     'mixed': cartesian_prod( {'a':[(1,2),(2,3),{'c':(3,2)}],'b':{'c':[(2,1), (2,2)]}}, 
                                        ( ('a',None), ('b',None), (0,'key_error'), ('c','key_error') ) )
    }
    @pytest.mark.parametrize('contents, key, exception', utils.unpack_test_cases(HASH_INDEXING_TEST_CASES), 
                                                         ids=utils.unpack_test_ids(HASH_INDEXING_TEST_CASES),
                                                         indirect=['contents'])
    def test_indexing_with_hash(self, contents, key, exception):

        arraytainer = self.container_class(contents)

        if exception is None:
            result = arraytainer[key]
            expected = contents[key]
            if isinstance(result, self.container_class):
                utils.assert_equal_values(result.unpacked, expected)
                utils.assert_same_types(result, self.container_class(expected))
            else:
                utils.assert_equal_values(result, expected)
        else:
            self.assert_exception(lambda x, key: x[key], exception, arraytainer, key)

    SLICE_INDEXING_TEST_CASES = \
    {'simple_dict': cartesian_prod( {'a':(6,),'b':(7,)}, ( (slice(1),None), (slice(1,5),None), (slice(1,10,2),None), 
                                                            ((slice(1,5,2), slice(1,5,2)),'key_error') ) ),
     'simple_list': cartesian_prod( [(3,3),(2,2)], ( (slice(1),None), (slice(0,2),None), ((slice(0),slice(0,2)),None) ) ),
     'nested_list': cartesian_prod( [[(2,2,2),(2,)],(2,1)], ( (slice(0,2),None), ((slice(0,0),slice(0,0)),'key_error') ) ),
     'nested_dict': cartesian_prod( {'a':(2,1),'b':{'c':(2,2),'d':(2,)},'c':(1,1)}, 
                                            ( (slice(0,2), None), ((slice(0,0),slice(0,0)),'key_error') ) ),
     'mixed': cartesian_prod( {'a':[(1,2),(3,),{'c':(3,2)}],'b':{'c':[(2,1),(2,2)]}}, 
                                            ( (slice(0,2),None), ((slice(0,0),slice(0,0)),'key_error') ) )
    }
    @pytest.mark.parametrize('contents, slice_val, exception', utils.unpack_test_cases(SLICE_INDEXING_TEST_CASES),
                                                              ids=utils.unpack_test_ids(SLICE_INDEXING_TEST_CASES),
                                                              indirect=['contents'])
    def test_indexing_with_slice(self, contents, slice_val, exception):

        arraytainer = self.container_class(contents)
        index_func = lambda contents : contents[slice_val]

        if exception is None:
            result = arraytainer[slice_val]
            expected, _ = utils.apply_func_to_contents(contents, func=index_func, throw_exception=True)
            utils.assert_equal_values(result.unpacked, expected)
            utils.assert_same_types(result, self.container_class(expected))
        else:
            self.assert_exception(lambda x, key: x[key], exception, arraytainer, slice_val)

    ARRAY_INDEX = ( ([0,-1],None), ([0,0,0],None), ([[0,-1,0],[-1,0,-1]],None),
                    ([True,False], None), ([[True,False],[False,True]], None),
                    ([[True],[False]], 'key_error') )
    INDEX_ARRAY_TEST_CASES = \
    {'dict': cartesian_prod( {'a':(2,2),'b':(2,2)}, ARRAY_INDEX ),
     'list': cartesian_prod( [(2,2),(2,2)], ARRAY_INDEX ),
     'nested_list': cartesian_prod( [[[(2,2)]],[(2,2),(2,2)]], ARRAY_INDEX ),
     'nested_dict': cartesian_prod( {'a':{'c':{'a':(2,2)},'d':(2,2)},'b':{'b':(2,2),'c':(2,2)}}, ARRAY_INDEX ),
     'mixed_a': cartesian_prod( [{'a':[(2,2),(2,2)],'c':(2,2)},{'b':(2,2),'c':{'d':(2,2)}}], ARRAY_INDEX ),
     'mixed_b': cartesian_prod( {'a':[(2,2),{'c':(2,2)}], 'b':[[(2,2),(2,2)],(2,2)]}, ARRAY_INDEX )
    }
    @pytest.mark.parametrize('contents, array, exception', utils.unpack_test_cases(INDEX_ARRAY_TEST_CASES), 
                                                           ids=utils.unpack_test_ids(INDEX_ARRAY_TEST_CASES),
                                                           indirect=['contents', 'array'])
    def test_indexing_with_array(self, contents, array, exception):
        arraytainer = self.container_class(contents)
        index_func = lambda contents : contents[array]
        if exception is None:
            expected, _ = utils.apply_func_to_contents(contents, func=index_func, throw_exception=True)
            result = arraytainer[array]
            utils.assert_equal_values(result.unpacked, expected)
            utils.assert_same_types(result, self.container_class(expected))
        else:
            self.assert_exception(index_func, exception, arraytainer)

    # We directly create arrays here with np.array to correctly speciy array shapes in content indices:
    INDEX_CONTENTS_TEST_CASES = \
    {'dict': cartesian_prod({'a':(2,2),'b':(3,3)}, ( ( {'a':[0,-1,-1],'b':[0,0]}, None ),
                                                    ( {'a':np.array([True,False]),'b':np.array([True,False,True])}, None ),
                                                    ( {'a':np.array(2*[[True,False]]),'b':np.array(3*[[True,False,True]])}, None ),
                                                    ( {'a':np.array([[True,False]]),'b':np.array([True,False,True])}, 'key_error' ),
                                                ( {'a':{'c':np.array([[True,False]])},'b':np.array([True,False,True])}, 'key_error' )  ) ),
     'list': cartesian_prod([(2,2),(3,3)], ( ( [np.array([0,-1,-1]),np.array([0,0])], None ),
                                             ( [np.array([True,False]),np.array([True,False,True])], None ),
                                             ( [np.array(2*[[True,False]]),np.array(3*[[True,False,True]])], None ),
                                             ( [np.array([[True,False]]),np.array([True,False,True])], 'key_error' ) ) ),  
     'nested_dict': cartesian_prod({'a':{'c':(2,2),'e':{'d':(2,2)}},'b':{'b':(3,3),'e':(3,3)}}, 
                                   ( ( {'a':np.array([0,-1,-1]),'b':np.array([0,0])}, None ),
                                     ( {'a':np.array(2*[[True,False]]),'b':np.array(3*[[True,False,True]])}, None ),
                                     ( {'a':np.array(2*[[True,False]]),'b':np.array(3*[[True,False,True]])}, None ),
                    ( {'a':{'c':np.array(2*[[True,False]]),'e':np.array(2*[[False,True]])},'b':np.array(3*[[True,False,True]])}, None ),
                    ( {'a':{'c':np.array(2*[[True,False]]),'e':np.array(2*[[False,True]])},
                       'b':{'b':np.array(3*[[True,False,True]]),'e':np.array([True,False,True])}}, None ) ) ),
     'mixed': cartesian_prod( {'a':[{'c':(2,2)},[(2,2),(2,2)],(2,2)],'b':[(3,3),{'c':(3,3),'d':(3,3)}]}, 
                              ( ( {'a':np.array([False,True]),'b':np.array([True,False,True])}, None ),
                                ( {'a':np.array([[False,True],[True,False]]), 
                                  'b':[np.array([True,False,True]),np.array([True,True,False])]}, None ),
                                ( {'a':np.array([[False,True],[True,True]]), 
                                  'b':[np.array([True,False,True]), {'c': np.array([True,False,False]), 
                                  'd': np.array([True,False,True])}]}, None ),
                                ({'a':[np.array([False,True])],'b':np.array([True,False,True])}, 'key_error') ) )
    }
    @pytest.mark.parametrize('contents, index_contents, exception', utils.unpack_test_cases(INDEX_CONTENTS_TEST_CASES), 
                                                                    ids=utils.unpack_test_ids(INDEX_CONTENTS_TEST_CASES),
                                                                    indirect=['contents'])
    def test_indexing_with_arraytainer(self, contents, index_contents, exception):

        arraytainer = self.container_class(contents)
        index_arraytainer = self.container_class(index_contents)

        # For case of Jaxtainer, check that index contents correctly converted to Jax arrays:
        if 'jax' in self.container_class.__name__.lower():
            array_list = utils.get_list_of_arrays(index_arraytainer.unpacked)
            assert all([isinstance(x,self.expected_array_types) for x in array_list])
            # If passed assertion, replace index_contents with unpacked arraytainer (since this will
            # contain Jax arrays instead of Numpy arrays):
            index_contents = index_arraytainer.unpacked

        index_func = lambda x, idx : x[idx]

        if exception is None:
            
            index_args_fun = lambda args, idx : (args[0][idx],) if not utils.is_array(args[0]) else args
            expected, _ =  utils.apply_func_to_contents(contents, func=index_func, args=(index_contents,), 
                                                        index_args_fun=index_args_fun, throw_exception=True)
            # First try indexing with contents NOT converted to arraytainer - should work same as arraytainer:
            result_1 = arraytainer[index_contents]
            utils.assert_equal_values(expected, result_1.unpacked)
            utils.assert_same_types(self.container_class(expected), result_1)
            # Next try indexing with arraytainer version of contents:
            result_2 = arraytainer[index_arraytainer]
            utils.assert_equal_values(expected, result_2.unpacked)
            utils.assert_same_types(self.container_class(expected), result_2)
        else:
            self.assert_exception(index_func, exception, arraytainer, index_contents)
            self.assert_exception(index_func, exception, arraytainer, index_arraytainer)