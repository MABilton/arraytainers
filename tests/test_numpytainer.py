import numpy as np
from arraytainers import Numpytainer

from test_class import ArraytainerTests

class TestNumpytainer(ArraytainerTests):

    container_class = Numpytainer
    array = np.array
