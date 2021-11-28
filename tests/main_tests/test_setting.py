    # def test_set_with_hash(self, hash_contents_keys_values):
    #     arraytainer = self.container_class(contents)
        
    #     with exception:
    #         arraytainer[key] = new_val

    #     assert arraytainer[key] == self.array(new_val)

    # def test_set_nonarraytainer_with_array_or_slice(self, arrays_contents_keys_nonarraytainervalues):
        
    #     arraytainer = self.container_class(contents)

    #     expected, exception = apply_func_to_contents(contents, args=(array_or_slice, new_val), func=setter_func)

    #     with exception:
    #         arraytainer[array_or_slice] = new_val

    #     assert arraytainer[array_or_slice] == self.container_class(expected)

    # def test_set_nonarraytainer_with_arraytainer(self, arraytainers_contents_keys_nonarraytainervalues):
        
    #     arraytainer = self.container_class(contents)
    #     arraytainer_key = self.container_class(key_contents)

    #     with exception:
    #         to_set = self.container_class(new_val) if val_is_container else new_val
    #         arraytainer[arraytainer_key] = to_set

    #     def index_args(args, idx):
    #         key_contents, new_val = args
    #         new_val_idxed = new_val[idx] if val_is_container else new_val
    #         return (key_contents[idx], new_val_idxed)

    #     expected = apply_func_to_contents(contents, args=(key_contents,new_val), func=setter_func, index_args=index_args)

    #     assert arraytainer[arraytainer_key] == self.container_class(expected)

    # def test_set_arraytainer_with_array_or_slice(self, arrays_contents_keys_arraytainervalues):
        
    #     arraytainer = self.container_class(contents)
    #     val_arraytainer = self.container_class(val_contents)

    #     def index_args(args, idx):
    #         array_or_slice, new_val = args
    #         return (array_or_slice, new_val[idx])

    #     expected, exception = 
    #         apply_func_to_contents(contents, args=(array_or_slice, val_contents), func=setter_func, index_args=index_args)

    #     with exception:
    #         arraytainer[array_or_slice] = val_arraytainer

    #     assert arraytainer[array_or_slice] == self.container_class(expected)

    # def test_set_arraytainer_with_arraytainer(self, arraytainers_contents_keys_arraytainervalues):
        
    #     arraytainer = self.container_class(contents)
    #     key_arraytainer = self.container_class(key_contents)
    #     val_arraytainer = self.container_class(val_contents)

    #     def index_args(args, idx):
    #         key_contents, val_contents = args
    #         return (key_contents[idx], val_contents[idx])

    #     expected, exception = 
    #         apply_func_to_contents(contents, args=(key_contents, val_contents), func=setter_func, index_args=index_args)

    #     with exception:
    #         to_set = self.container_class(new_val) if val_is_container else new_val
    #         arraytainer[arraytainer_key] = to_set

    #     assert arraytainer[arraytainer_key] == self.container_class(expected)