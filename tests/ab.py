binary_arg_array_input = {'a': {'b': [np.array([[1,2],[3,4]])]}, 'c': [np.ones((2,2,2)), 0.5*np.ones((2,2))]}
array = np.array([1.5, np.pi])
binary_op_array_tests = [
    {'func': lambda x,y: x+y, 
     'expected': {'a': {'b': [np.array([[1,2],[3,4]])+array], np.array(np.array(1)+array}, 'c': [np.ones((2,2,2))+array, 0.5*np.ones((2,2))+array]}},
    {'func': lambda x,y: x-y, 
     'expected': {'a': {'b': [np.array([[1,2],[3,4]])-array, np.array(np.array(1)-array)]}, 'c': [np.ones((2,2,2))-array, 0.5*np.ones((2,2))-array]}},
    {'func': lambda x,y: x*y, 
     'expected': {'a': {'b': [np.array([[1,2],[3,4]])*array, np.array(np.array(1)*array)]}, 'c': [np.ones((2,2,2))*array, 0.5*np.ones((2,2))*array]}},
    {'func': lambda x,y: x/y, 
     'expected': {'a': {'b': [np.array([[1,2],[3,4]])/array, np.array(np.array(1)/array)]}, 'c': [np.ones((2,2,2))/array, 0.5*np.ones((2,2))/array]}},
    {'func': lambda x,y: x**y, 
     'expected': {'a': {'b': [np.array([[1,2],[3,4]])**array, np.array((np.array(1))**array)]}, 'c': [np.ones((2,2,2))**array, (0.5*np.ones((2,2)))**array]}},
    {'func': lambda x,y: y+x, 
     'expected': {'a': {'b': [array+np.array([[1,2],[3,4]]), np.array(array+np.array(1))]}, 'c': [array+np.ones((2,2,2)), array+0.5*np.ones((2,2))]}},
    {'func': lambda x,y: y-x, 
     'expected': {'a': {'b': [array-np.array([[1,2],[3,4]]), np.array(array-np.array(1))]}, 'c': [array-np.ones((2,2,2)), array-0.5*np.ones((2,2))]}},
    {'func': lambda x,y: y*x, 
     'expected': {'a': {'b': [array*np.array([[1,2],[3,4]]), np.array(array*np.array(1))]}, 'c': [array*np.ones((2,2,2)), array*0.5*np.ones((2,2))]}},
    {'func': lambda x,y: y/x, 
     'expected': {'a': {'b': [array/np.array([[1,2],[3,4]]), np.array(array/(np.array(1)))]}, 'c': [array/np.ones((2,2,2)), array/(0.5*np.ones((2,2)))]}},
    {'func': lambda x,y: y**x, 
     'expected': {'a': {'b': [array**np.array([[1,2],[3,4]]), np.array(array**(np.array(1)))]}, 'c': [array**np.ones((2,2,2)), array**(0.5*np.ones((2,2)))]}},
]