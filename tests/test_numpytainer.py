import numpy as np
from arraytainers import Numpytainer
from main_tests.test_class import ArraytainerTests

class TestNumpytainer(ArraytainerTests):

    container_class = Numpytainer
    array_constructor = np.array
    expected_array_types = (np.ndarray,)

    # self.array = np.array
    # self.array_types = (np.ndarray,)