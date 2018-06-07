import pandas as pd
import numpy as np


def discretise(data, bucketing_type, variable, n = None, quantiles = None, weights = None):

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

        bucketed_variable = quantile(data = data, 
                                     variable = variable,
                                     quantiles = quantiles)

    elif bucketing_type == 'weighted_quantile':

        raise ValueError('weighted_quantile bucketing_type not yet supported.')

    else:

        raise ValueError('unexpected value for bucketing_type %s' % bucketing_type)

    return(bucketed_variable)


def equal_width(variable, n):

    # see https://github.com/pandas-dev/pandas/issues/17047 for issues with include_lowest
    variable_cut = pd.cut(variable, n, include_lowest = True)

    variable_cut = add_null_category(variable_cut)

    return(variable_cut)



def equal_weight(data, variable, weights, n):

    if weights == None:

        df = data[[variable]].copy()

        df ['weights'] = 1

        weights = 'weights'

        df.sort_values(variable, inplace = True)

    else:

        df = data[[variable, weights]].sort_values(variable)

    # group by variable of interest to get weights and count by each data value
    summary = df.groupby(df[variable]).agg({weights: ['sum', 'count']})

    summary.columns = summary.columns.get_level_values(1)

    summary.reset_index(inplace = True)

    bin_weight = summary['sum'].sum() / n

    rows = []
    weighted_cut_points = []

    bucket_sum = 0

    for i in range(summary.shape[0]):
        
        bucket_sum += summary.iloc[i]['sum']
        
        if bucket_sum >= bin_weight:
            
            rows.append(i)
            
            weighted_cut_points.append(summary.iloc[i][variable])
            
            bucket_sum = 0
    
    # if the last value that was added was not from the last row
    if (weighted_cut_points[len(weighted_cut_points)-1]) != summary.iloc[i][variable]:
        
        rows.append(i)
        
        weighted_cut_points.append(summary.iloc[i][variable])

    variable_cut = pd.cut(data[variable], weighted_cut_points, include_lowest = True)
    
    variable_cut = add_null_category(variable_cut)

    return(variable_cut)




def quantile(data, variable, quantiles):

    quantiles = np.array(quantiles)

    quantiles = np.unique(np.sort(np.append(quantiles, [0, 1])))

    quantile_values = data[variable].quantile(quantiles)

    variable_cut = pd.cut(data[variable], np.unique(quantile_values), include_lowest = True)

    variable_cut = add_null_category(variable_cut)

    return(variable_cut)    


def add_null_category(categorical_variable):

    cat = categorical_variable.cat.add_categories(['Null'])

    cat.fillna('Null', inplace = True)

    return(cat)




