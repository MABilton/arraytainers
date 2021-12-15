    
import pytest
import operator
import numpy as np
from copy import deepcopy
from itertools import product
from . import utils
from .utils import cartesian_prod

class SetMixin:
    # Must place (2,3) inside a tuple for flattening purposes:
    HASH_SET_VALUES = ( (2,3), [(2,3)], [[(2,1)],[(2,2)]], {'a':(2,2)}, {'a':{'b':(2,2),'c':(2,1)}}, 
                        [{'a':[(1,),(2,)],'b':[(2,),(3,)]}, {'c':[(2,1),{'c':(2,2)}]}],
                        {'a':[(2,2),(2,1)], 'b':[(2,1),{'b':[(2,),(3,)]}]} )
    INDEPENDENCE_TEST_CASES = {
    'list': cartesian_prod( [[(2,3),[(2,1)]],(2,1),[(1,),(2,)]], HASH_SET_VALUES, ( (0,), (0,1), (2,1)), flatten=False),
    'dict': cartesian_prod( {'a':{'a':(2,3),'b':{'c':(2,1)}},'b':{'a':(1,2),'b':(2,2)}}, HASH_SET_VALUES,
                             ( ('a',), ('b',), ('a','a'), ('a','b'), ('a','b','c'), ('b','a') ), flatten=False),
    'mixed': cartesian_prod( [{'a':[(2,1),(1,2)],'b':(2,1)},{'b':{'c':(2,1)}}], HASH_SET_VALUES,
                              ( (0,), ((0,'a',1)), (1,'b'), (1,'b','c') ), flatten=False) 
    }
    INDEPENDENCE_TEST_CASES = utils.group_first_n_params(INDEPENDENCE_TEST_CASES, n=2)
    @pytest.mark.parametrize('contents_list, key_iterable', utils.unpack_test_cases(INDEPENDENCE_TEST_CASES), 
                                                            ids=utils.unpack_test_ids(INDEPENDENCE_TEST_CASES),
                                                            indirect=['contents_list'])
    def test_independence(self, contents_list, key_iterable):
        contents, new_val = contents_list
        contents_copy = deepcopy(contents)
        arraytainer = self.container_class(contents_copy)
        # After changing the value of contents, arraytainer should remain the same - ie the values
        # stored in the arraytainer should be independent of the initial contents used to create it
        utils.set_contents_item(contents_copy, key_iterable, new_val)
        utils.assert_equal_values(arraytainer.unpacked, contents)
    
    HASH_SET_VALUES_TUPLES = tuple(tuple([x]) if isinstance(x, tuple) else x for x in HASH_SET_VALUES)
    SET_METHOD_TEST_CASES = {
        'list': cartesian_prod( [[(2,3),(3,2)],(2,2),[(2,1),[(2,2),(3,2)]]],
                                HASH_SET_VALUES_TUPLES,
                              ( ((0,),None), ((0,1),None), ((1,),None), ((2,1),None), ((2,1,1),None),
                                ((3,), 'key_error'), ((2,0,0),AttributeError), ((), 'key_error') ) ),
        'dict': cartesian_prod( {'a':{'a':(2,1),'c':(2,2)},'b':{'d':(3,2),'b':(2,2)}},
                                HASH_SET_VALUES_TUPLES,
                              ( (('a',), None), (('b',), None), (('c',), None), (('a','a'),None), (('a','e'),None),
                                ((0,), None), ((0,0), 'key_error'),  (('a','a','a'), AttributeError), ((), 'key_error') ) ),
        'mixed': cartesian_prod( [{'a':[(2,2),{'c':(2,1)},(1,2)],'b':(2,1)},{'b':(2,1),'c':[(2,2),(1,2)]}], 
                                 HASH_SET_VALUES_TUPLES,
                              ( ((0,), None), ((1,), None), ((0,'a'), None), ((0,'a',1,'c'), None), ((1,'c',0), None),
                                ((0,'b'), None), ((0,'a',3), 'key_error'), ((), 'key_error') ) )
    }
    SET_METHOD_TEST_CASES = utils.group_first_n_params(SET_METHOD_TEST_CASES, n=2)
    @pytest.mark.parametrize('contents_list, key_iterable, exception', utils.unpack_test_cases(SET_METHOD_TEST_CASES), 
                                                                       ids=utils.unpack_test_ids(SET_METHOD_TEST_CASES), 
                                                                       indirect=['contents_list'])
    def test_set_and_copy_method(self, contents_list, key_iterable, exception):
        
        contents, new_val_contents = contents_list
        
        if exception is None:
          contents_copy = deepcopy(contents)
          utils.set_contents_item(contents_copy, key_iterable, new_val_contents)

          # Test two independent sets of conditions:
          #    1. key_iterables passed as unpacked tuple vs passed as packed tuple
          #    2. new_val is an arraytainer vs new_val is not an arraytainer 
          # If contents is just a lone array, do NOT test situation where new_val_contents is converted
          # to arraytainer, since doing so will place array in a list:
          val_is_container_tests = (False, True) if not utils.is_array(new_val_contents) else (False,)
          for unpack_keys, val_is_container in product((False, True), val_is_container_tests):
            arraytainer = self.container_class(contents)
            arraytainer_copy = arraytainer.copy()
            new_val = self.container_class(new_val_contents) if val_is_container else new_val_contents
            if unpack_keys:
              arraytainer.set(new_val, *key_iterable)
            else:
              arraytainer.set(new_val, key_iterable)
            utils.assert_equal_values(arraytainer.unpacked, contents_copy)
            utils.assert_same_types(arraytainer, self.container_class(contents_copy))
            # Esure setting has not affected copy of arraytainer:
            utils.assert_equal_values(arraytainer_copy.unpacked, contents)
        
        else:
          arraytainer = self.container_class(contents)
          # Throw exception when new_val is a list/dict and when container
          for new_val in (new_val_contents, self.container_class(new_val_contents)):
            self.assert_exception(lambda arraytainer, new_val, key_iterable : arraytainer.set(new_val, *key_iterable),
                                  exception, arraytainer, new_val, key_iterable)

    # APPEND_METHOD_TEST_CASES
    # def test_append_method(self, contents, key_iterable, new_value, exception):
    #     pass

    # # def test_increment_with_arraytainer(self, std_contents, val, func):
    # SET_WITH_HASH_ARRAYTAINERS = { 'arraytainer_1': {'a':[(2,2),[(2,),(3,)]],'b':[(2,2),{'c':(2,1),'d':(3,1)}]},
    #                                'arraytainer_2': [{'a':[(2,1),(2,2)],'b':(2,2)},{'a':(2,1),'b':[(3,2),(2,1)]}] }
    # SET_WITH_HASH_ARRAYS = {'array_1': (2,2), 'array_2': ()}
    # @pytest.mark.parametrize('contents', SET_WITH_HASH_ARRAYTAINERS.values(), ids=SET_WITH_HASH_ARRAYTAINERS.keys(), indirect=['contents'])
    # @pytest.mark.parametrize('set_val', [*SET_WITH_HASH_ARRAYS.values(), *SET_WITH_HASH_ARRAYTAINERS.values()], 
    #                                       ids=[*SET_WITH_HASH_ARRAYS.keys(), *SET_WITH_HASH_ARRAYTAINERS.keys()])
    # @pytest.mark.parametrize('key', SET_WITH_HASH_KEYS)
    # def test_set_with_hash(self, contents, key, set_val):

    #     arraytainer = self.container_class(contents)

    #     new_val_is_container = not isinstance(set_val, tuple)
    #     if new_val_is_container:
    #         val = self.container_class(set_val)
    #     else:
    #         val = self.array_constructor(set_val)

    #     arraytainer[key] = val
    #     utils.assert_equal_values(arraytainer[key].unpacked, val)
    #     utils.assert_same_types(arraytainer[key], self.container_class(set_val))

    #     # In case of setting with a container, setting with the 'raw', non-containerised contents
    #     # should also work:
    #     if new_val_is_container:
    #         arraytainer = self.container_class(contents)
    #         arraytainer[key] = set_val
    #         utils.assert_equal_values(arraytainer[key].unpacked, val)
    #         utils.assert_same_types(arraytainer[key], self.container_class(set_val))

        
#     @pytest.mark.parametrize('array', utils.unpack_test_cases(ARRAY_TEST_CASES), 
#                                       ids=utils.unpack_test_ids(ARRAY_TEST_CASES),
#                                       indirect = ['array])
#     def test_increment_with_array(self, contents, array, func, exception)

#     def test_increment_with_arraytainer(self, contents, func, exception)

#     def test_set_with_hash(self, contents, key, exception):
#         arraytainer = self.container_class(contents)
        
#         with exception:
#             arraytainer[key] = new_val

#         assert arraytainer[key] == self.array(new_val)

#     def test_set_with_array

#     # def test_set_nonarraytainer_with_array_or_slice(self, arrays_contents_keys_nonarraytainervalues):
        
#     #     arraytainer = self.container_class(contents)

#     #     expected, exception = apply_func_to_contents(contents, args=(array_or_slice, new_val), func=setter_func)

#     #     with exception:
#     #         arraytainer[array_or_slice] = new_val

#     #     assert arraytainer[array_or_slice] == self.container_class(expected)

#     # def test_set_nonarraytainer_with_arraytainer(self, arraytainers_contents_keys_nonarraytainervalues):
        
#     #     arraytainer = self.container_class(contents)
#     #     arraytainer_key = self.container_class(key_contents)

#     #     with exception:
#     #         to_set = self.container_class(new_val) if val_is_container else new_val
#     #         arraytainer[arraytainer_key] = to_set

#     #     def index_args(args, idx):
#     #         key_contents, new_val = args
#     #         new_val_idxed = new_val[idx] if val_is_container else new_val
#     #         return (key_contents[idx], new_val_idxed)

#     #     expected = apply_func_to_contents(contents, args=(key_contents,new_val), func=setter_func, index_args=index_args)

#     #     assert arraytainer[arraytainer_key] == self.container_class(expected)

#     # def test_set_arraytainer_with_array_or_slice(self, arrays_contents_keys_arraytainervalues):
        
#     #     arraytainer = self.container_class(contents)
#     #     val_arraytainer = self.container_class(val_contents)

#     #     def index_args(args, idx):
#     #         array_or_slice, new_val = args
#     #         return (array_or_slice, new_val[idx])

#     #     expected, exception = 
#     #         apply_func_to_contents(contents, args=(array_or_slice, val_contents), func=setter_func, index_args=index_args)

#     #     with exception:
#     #         arraytainer[array_or_slice] = val_arraytainer

#     #     assert arraytainer[array_or_slice] == self.container_class(expected)

#     # def test_set_arraytainer_with_arraytainer(self, arraytainers_contents_keys_arraytainervalues):
        
#     #     arraytainer = self.container_class(contents)
#     #     key_arraytainer = self.container_class(key_contents)
#     #     val_arraytainer = self.container_class(val_contents)

#     #     def index_args(args, idx):
#     #         key_contents, val_contents = args
#     #         return (key_contents[idx], val_contents[idx])

#     #     expected, exception = 
#     #         apply_func_to_contents(contents, args=(key_contents, val_contents), func=setter_func, index_args=index_args)

#     #     with exception:
#     #         to_set = self.container_class(new_val) if val_is_container else new_val
#     #         arraytainer[arraytainer_key] = to_set

#     #     assert arraytainer[arraytainer_key] == self.container_class(expected)