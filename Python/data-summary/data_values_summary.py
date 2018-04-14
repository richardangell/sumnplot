import pandas as pd


def data_values_summary(df, columns = None, max_values = 50, summary_values = 5, top_mid_bottom = 5):
    
    if columns == None:
        
        columns = df.columns.values
    
    columns_summary = [summarise_column(df, col, max_values, summary_values, top_mid_bottom) for col in columns]
    
    columns_summary_all = pd.concat(columns_summary, axis = 1)

    return(columns_summary_all)
    

def summarise_column(df, column, max_values, summary_values, top_mid_bottom):
    
    value_counts = df[column].value_counts().sort_index(ascending = True).reset_index()
    
    value_counts.columns = [column + ".value", column + ".count"]
    
    value_counts_resize = resize_column_summary(value_counts, max_values, summary_values, top_mid_bottom)
    
    return(value_counts_resize)
    
    
def resize_column_summary(df, max_values, summary_values, top_mid_bottom):
    
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