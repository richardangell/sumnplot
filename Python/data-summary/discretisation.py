import pandas as pd



def discretise(data, bucketing_type, variable, n = None, weights = None):

    type_valid_options = ['equal_width', 'equal_weight', 'quantile', 'weighted_quantile']

    assert bucketing_type in type_valid_options, 'invalid bucketing_type; %s, valid; %s' % (type, type_valid_options)

    assert type(data) is pd.DataFrame, 'data is not a pandas DataFrame'

    assert variable in data.columns.values, 'variable; %s is not in data' % variable

    if weights is not None:

        assert weights in data.columns.values, 'weights; %s is not in data' % weights

    if bucketing_type == 'equal_width':

        bucketed_variable = equal_width(variable = data[variable],
                                        n = n)

    elif bucketing_type == 'equal_weight':

        bucketed_variable = equal_weight(data = data, 
                                         variable = variable,
                                         weights = weights,
                                         n = n)

    elif bucketing_type == 'quantile':

        raise ValueError('quantile bucketing_type not yet supported.')

    elif bucketing_type == 'weighted_quantile':

        raise ValueError('weighted_quantile bucketing_type not yet supported.')

    else:

        raise ValueError('unexpected value for bucketing_type %s' % bucketing_type)

    return(bucketed_variable)


def equal_width(variable, n):

    print('aa')
    
    variable_cut = pd.cut(variable, n, include_lowest = True)

    variable_cut = add_null_category(variable_cut)

    return(variable_cut)



def equal_weight(data, variable, weights, n):

    print('bb')

    df = data[[variable, weights]].sort_values(variable)

    # group by variable of interest to get weights and count by each data value
    summary = df.groupby(df[variable]).agg({weights: ['sum', 'count']})

    summary.columns = summary.columns.get_level_values(1)

    summary.reset_index(inplace = True)

    summary[variable] = summary[variable].astype(str)

    total_count = summary['count'].sum()

    # if the total record count does not equal the number of rows from df
    # then create a new row to append containing the sum and count of null rows
    # this is required because pandas removes null rows from the groupby 
    # see https://github.com/pandas-dev/pandas/issues/3729 for more info
    if total_count != df.shape[0]:
        
        null_row = pd.DataFrame(data = {variable: 'Null', 
                                        'sum': df[weights].sum() - summary['sum'].sum(), 
                                        'count': df.shape[0] - total_count}, 
                                index = [summary.shape[0]])

        summary = summary.append(null_row)

    summary['cumsum_weight'] = summary['sum'].cumsum(skipna = False)

    summary['cumsum_weight_pct'] = summary.cumsum_weight / summary['sum'].sum() 

    summary['weighted_bucket'] = summary.cumsum_weight_pct / (1 / n)

    

    return(2)



def add_null_category(categorical_variable):

    cat = categorical_variable.cat.add_categories(['Null'])

    cat.fillna('Null', inplace = True)

    return(cat)




