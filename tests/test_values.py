
import numpy as np

def create_test_values():

    # Define arrays:
    list_contents = {f'{dim}d': [coeff*np.ones(dim*(dim,)) for coeff, size in zip(coefficients, sizes)}
                    for dim in array_dims_to_check]
    list_contents['empty'] = []


    dict_contents = {f'{dim}d': {f'{str(size)}': coeff*np.ones(dim*(dim,)) for coeff, size in zip(coefficients, sizes)}
                    for dim in array_dims_to_check}
    dict_contents['empty'] = {}

    nested_contents = {'empty':,
                       'dict_list':,
                       'list_dict':,
                       'dict_dict':,
                       'list_list':,
                       'list_dict_list':,
                       'dict_list_dict':,


    }

    return 