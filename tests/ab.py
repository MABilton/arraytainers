list_arrays_arraytainer_tests = [
    {'contents': {'a': 1, 'b': jnp.ones((3,3,3)), 'c': jnp.array([[1,2]])},
     'expected': [jnp.array(1), jnp.ones((3,3,3)), jnp.array([[1,2]])] },
    {'contents': [1, jnp.ones((3,3,3)), jnp.array([[1,2]])],
     'expected': [jnp.array(1), jnp.ones((3,3,3)), jnp.array([[1,2]])] }, 
    {'contents': {'a': {'c': [jnp.ones((3,3,3)), jnp.array([[9,10]])]}, 1: [{'a': jnp.array([[11,12],[13,14]])}, jnp.array(15)]},
     'expected': [jnp.ones((3,3,3)), jnp.array([[9,10]]), jnp.array([[11,12],[13,14]]), jnp.array(15)] },
    {'contents': {'a': [{'c': True}, jnp.ones((2,2))], 'b': {'d':[jnp.array([True]), 1]}},
     'expected': [jnp.array(True), jnp.ones((2,2)), jnp.array([True]), jnp.array(1)] }
]