








def discretise(data, variable, weights, type):

    type_valid_options = ['equal_width', 'equal_weight', 'quantile', 'weighted_quantile']

    assert type in type_valid_options, 'invalid type supplied, got %s; valid; %s' % (type, type_valid_options)






