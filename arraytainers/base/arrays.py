import numpy as np
import functools
    
class Mixin:
    
    def array(self, x):
        if self.is_container(x):
            return x
        # elif self.is_array(x):
        #     # Coverting from Jax to Numpy array can change shape:
        #     before_shape = x.shape
        #     x = self.array_class(x)
        #     return x.reshape(before_shape)
        else:
            return self.array_class(x)

    # Transpose all arrays in arraytainer:
    @property
    def T(self):
        return np.transpose(self)

    # Returns 'True' only if all values in arraytainer are true:
    def all(self):
        for key in self.keys():
            if self.contents[key].all():
                continue
            else:
                return False
        return True

    # Returns 'True' if any value in arraytainer is true:
    def any(self):
        for key in self.keys():
            if self.contents[key].any():
                return True
        return False

    def sum(self, elements=True):
        return self.sum_elements() if elements else self.sum_arrays()

    # Sums up all elements stored in arraytainer:
    def sum_elements(self):
        
        if list(self.keys()):
            sum_results = functools.reduce(lambda x,y: x+y, [val for val in self.values()])
        else:
            sum_results = 0

        if not self.is_container(sum_results):
            sum_results =   self.__class__(sum_results)

        return sum_results

    # Adds all arrays stored in arraytainer together:
    def sum_arrays(self):
        to_sum = self.list_arrays()
        return self.array(sum(to_sum))

    # Shape methods: 
    def get_shapes(self, return_arraytainer=True):
        shapes = [self[key].get_shapes(return_arraytainer=False) if self.is_container(self[key]) 
                  else self[key].shape for key in self.keys()]
        if self._type is dict:
            shapes = dict(zip(self.keys(), shapes))
        if return_arraytainer:
            shapes = self.__class__(shapes, greedy_array_conversion=True)
        return shapes

    @property
    def shape(self):
        return self.get_shapes()

    def reshape(self, new_shapes):
        return np.reshape(self, new_shapes)

    def flatten(self):
        return np.ravel(self)