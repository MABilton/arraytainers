import numpy as np
from arraytainers import Numpytainer

from main_tests.test_class import ArraytainerTests

class TestNumpytainer(ArraytainerTests):

    container_class = Numpytainer
    array = np.array
    array_types = (np.ndarray,)