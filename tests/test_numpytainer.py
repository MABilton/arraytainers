import numpy as np
from arraytainers import Numpytainer

from .test_class import ArraytainerTests

class NumpytainerTests(ArraytainerTests):
    
    self.container_class = Numpytainer
    self.array = np.array
