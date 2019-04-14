import pandas as pd



def data_values_summary(df, columns = None, max_values = 50, summary_values = 5, top_mid_bottom = 5):
    '''Function to produce summaries of values in a DataFrame.'''

    if not isinstance(df, pd.DataFrame):

        raise TypeError('df should be a pandas DataFrame')

    if columns is None:
        
        columns = df.columns.values

    else:

        if not isinstance(columns, list):

            raise TypeError('columns should be a list (of columns in df)')
    
        for col in columns:

            if not col in df.columns.values:

                raise ValueError(str(col) + ' not in df')

    if not isinstance(max_values, int):

        raise TypeError('max_values should be an int')

    if not max_values > 0:

        raise ValueError('max_values must be greater than 0')

    if not isinstance(summary_values, int):

        raise TypeError('summary_values should be an int')

    if not summary_values > 0:

        raise ValueError('summary_values must be greater than 0')

    if not isinstance(top_mid_bottom, int):

        raise TypeError('top_mid_bottom should be an int')

    if not top_mid_bottom > 0:

        raise ValueError('top_mid_bottom must be greater than 0')

    columns_summary = [
        summarise_column(df, col, max_values, summary_values, top_mid_bottom) for col in columns
    ]
    
    columns_summary_all = pd.concat(columns_summary, axis = 1)

    return(columns_summary_all)
    


def summarise_column(df, column, max_values, summary_values, top_mid_bottom):
    '''Function to return value_counts for a sinlge column in df resized to max_values rows.'''

    value_counts = df[column].value_counts(dropna = False).sort_index(ascending = True).reset_index()
    
    value_counts.columns = [column + "_value", column + "_count"]
    
    value_counts_resize = resize_column_summary(value_counts, max_values, summary_values, top_mid_bottom)
    
    if not value_counts_resize.shape[0] == max_values:

        raise ValueError('unexpected shape for value_counts_resize for column; ' + column)

    return(value_counts_resize)
    

    
def resize_column_summary(df, max_values, summary_values, top_mid_bottom):
    '''Function to resize the output the results of value_counts() to be max_values rows.

    If n (number rows of df) < max_values then df is padded with rows containing None. Otherwise
    if n > max_values then the first, middle and last top_mid_bottom rows are selected and similarly
    padded.
    '''

    n = df.shape[0]
    
    if n == max_values:
        
        return(df.reset_index(drop = True))
    
    elif n < max_values:
        
        extra_rows = max_values - n
        
        append_rows = pd.DataFrame({df.columns[0]: extra_rows * [None],
                                    df.columns[1]: extra_rows * [None]})
        
        return(df.append(append_rows).reset_index(drop = True))
        
    else:
        
        select_rows = df.loc[0:(summary_values - 1)]
    
        append_rows = pd.DataFrame({df.columns[0]: 1 * [None],
                                    df.columns[1]: 1 * [None]})
    
        select_rows = select_rows.append(append_rows)
        
        mid_row = n // 2
        
        below_mid_row = mid_row - (summary_values // 2)
        
        middle_rows = df.loc[below_mid_row:(below_mid_row + summary_values)]
        
        select_rows = select_rows.append(middle_rows)
        
        select_rows = select_rows.append(append_rows)
        
        top_rows = df.loc[(n - summary_values):]
        
        select_rows = select_rows.append(top_rows)
        
        extra_rows = max_values - select_rows.shape[0]
    
        if extra_rows > 0:
            
            append_rows = pd.DataFrame({df.columns[0]: extra_rows * [None],
                                        df.columns[1]: extra_rows * [None]})
            
            select_rows = select_rows.append(append_rows)
    
        return(select_rows.reset_index(drop = True))



