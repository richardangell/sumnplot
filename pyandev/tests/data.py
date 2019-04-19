import pandas as pd
import numpy as np
from sklearn.datasets import load_boston



def load_boston_df(insert_missings = False, add_weights_column = False, add_dummy_predictions = False, add_dummy_predictions2 = False):
    '''Load boston dataset from sklearn and prepare it as a pandas DataFrame.'''

    if not isinstance(insert_missings, bool):

        raise TypeError('insert_missings must be bool')

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





