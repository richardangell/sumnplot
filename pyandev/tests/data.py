import pandas as pd
import numpy as np
from sklearn.datasets import load_boston



def load_boston_df(insert_missings = False):
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

    return boston_df





