from contextlib import contextmanager
import numpy as np


def create_contents(keys)


@pytest.fixture(scope="function")
def contents(coeffs, sizes, dims, keys):

    if keys is not None:
        contents = {f'{dim}d': [coeff*np.ones(dim*(dim,)) for coeff, size in zip(coeffs, sizes)}
                    for dim in dims]}
    else:
        contents = {f'{dim}d': {f'{key}': coeff*np.ones(dim*(dim,)) for key, coeff, size in zip(keys, coeffs, sizes)}
                    for dim in array_dims_to_check}

    yield contents

@pytest.fixture(scope="function")
def contents_and_idx(contents, idx_type, idx_is_valid, container_class):
    
    if idx_type is 'hash':
        if idx_is_valid:
            idx = 0 if isinstance(contents, list) else contents.keys()[0] 
        else:
            # Guaranteed to throw a key error since dict is unhashable:
            idx = {}

    elif idx_type is 'array':
        if idx_is_valid:
            
        else:

    elif idx_type is 'slice':
        if idx_is_valid:

        else:

    elif idx_type is 'container':
        if idx_is_valid:

        else:

    yield (contents, idx)

@pytest.fixture(scope="function")
def contents_idx_and_value(contents_and_idx, val_type, val_is_valid):



    yield (*contents_and_idx, new_val)