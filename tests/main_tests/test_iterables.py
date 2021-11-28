import pytest

from .utils import assert_equal_values

KEY_CHECKING = \
{'tuple_of_ints_1': ({(1,2): (1,)}, 'key'),
'tuple_of_ints_2': ({('1',2):(1,), (1,2):(1,)}, 'key'),
'valid_key': ({('1',2):(1,), (1,'2'):(1,)}, None)}

class IterableMixin:

    @pytest.mark.parametrize('contents, exception', KEY_CHECKING.values(), ids=KEY_CHECKING.keys())
    def test_key_checking(self, contents, exception):
        if exception is None:
            self.container_class(contents)
        else:
            self.assert_exception(self.container_class, exception, contents)

    def test_unpacking(self, std_contents):
        arraytainer = self.container_class(std_contents)
        assert_equal_values(arraytainer.unpacked, std_contents)

    def test_keys_values_items(self, std_contents):
        
        arraytainer = self.container_class(std_contents)

        if isinstance(std_contents, dict):
            values = tuple(std_contents.values())
            keys = tuple(std_contents.keys())
            items = tuple((key, val) for key, val in std_contents.items())            
        else:
            values = tuple(std_contents)
            keys = tuple(range(len(std_contents)))
            items = tuple((i, val) for i, val in enumerate(std_contents))
 
        assert set(arraytainer.keys()) == set(keys)
        assert_equal_values(arraytainer.values(unpacked=True), values)
        for result, expected in zip(arraytainer.items(unpacked=True), items):
            # Compare keys:
            assert result[0] == expected[0]
            # Compare arrays:
            assert_equal_values(result[1], expected[1])