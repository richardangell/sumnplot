

def data_values_summary(df, columns, summary_values = 20, top_mid_bottom = 5):

    for col in columns:
        
        # count each data value in specified column
        value_counts = df[col].value_counts().sort_index(ascending = True).reset_index()

        value_counts.columns = [col + ".value", col + ".count"]

        # number of unique values in column
        n = value_counts.shape[0]


