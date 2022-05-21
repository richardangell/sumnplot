import pandas as pd
import numpy as np
from sklearn.datasets import load_boston



def load_boston_df(insert_missings = False, 
                   add_weights_column = False, 
                   add_dummy_predictions = False, 
                   add_dummy_predictions2 = False):
    '''Load boston dataset from sklearn and prepare it as a pandas DataFrame.
    
    Parameters
    ----------
    insert_missings : bool, default = False
        Should nulls be inserted into the data? If True a random 10% of records (different across columns)
        are set to null.
        
    add_weights_column : bool, default = False
        Should a weights column be added to the data? If True random weights are added to each row. 
        Weights take values of 0 to 10/3, in steps of 1/3.

    add_dummy_predictions : bool, default = False
        Should a column of dummy predictions be added to the data? If True normal(0, 3) noise is added to
        the response as predictions.

    add_dummy_predictions2 : bool, default = False
        Should a column of dummy predictions be added to the data? If True normal(2, 3) noise is added to
        the response as predictions.        

    '''

    if not isinstance(insert_missings, bool):

        raise TypeError('insert_missings must be bool')

    if not isinstance(add_weights_column, bool):

        raise TypeError('add_weights_column must be bool')

    if not isinstance(add_dummy_predictions, bool):

        raise TypeError('add_dummy_predictions must be bool')

    if not isinstance(add_dummy_predictions2, bool):

        raise TypeError('add_dummy_predictions2 must be bool')

    boston = load_boston()

    if insert_missings:

        np.random.seed(1)

        missing_loc = np.random.random(boston['data'].shape)

        boston['data'][missing_loc < 0.1] = np.NaN

    boston_df = pd.DataFrame(boston['data'], columns = boston['feature_names'])

    boston_df['target'] = boston['target']

    if add_weights_column:

        np.random.seed(5)

        boston_df['weights'] = np.random.randint(low = 0, high = 10, size = boston_df.shape[0]) / 3

    if add_dummy_predictions:

        np.random.seed(9)

        boston_df['target_dummy_pred'] = boston_df['target'] + np.random.normal(0, 3, boston_df.shape[0])

    if add_dummy_predictions2:

        np.random.seed(13)

        boston_df['target_dummy_pred2'] = boston_df['target'] + np.random.normal(2, 3, boston_df.shape[0])

    return boston_df





