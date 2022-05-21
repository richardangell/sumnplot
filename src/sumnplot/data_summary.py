import pad.data_values_summary as dvs
import pandas as pd




class data_summary():
    """data summary class"""

    def __init__(self, df, columns):

        check_args(df = df, 
                   columns = columns)

        self.df = df
        self.columns = columns

    def data_values_summary(columns):

        check_columns(self.df, columns)

        self.data_values_summary = dvs.data_values_summary()




def check_args(df, columns):
    """Check all arguments to initialise data_summary class"""

    check_df(df)

    check_columns(df, columns)


def check_df(df):
    """Check if df argument is a pandas DataFrame"""

    assert isinstance(df, pd.DataFrame), "df is not a pandas DataFrame" 


def check_columns(df, columns):
    """Check columns is a list of column names, all present in df"""

    assert isinstance(columns, str) | isinstance(columns, list), "columns should be a list of names of columns in df to summarise"

    if isinstance(columns, list):

        columns_str = [isinstance(col, str) for col in columns]

        assert all(columns_str), "columns should be a list of strings"

    missing_columns = [col for col in columns if not col in df.columns.values]

    assert len(missing_columns) == 0, "the following columns are not in df; %s" % missing_columns






        