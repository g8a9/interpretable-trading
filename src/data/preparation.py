"""Utilities to prepare data for ML classifiers"""

from l3wrapper.l3wrapper import TRAIN_BIN
import pandas as pd
import numpy as np
import logging


logger = logging.getLogger(__name__)


CLASSES = [0, 1, 2]
TRAIN_SIZE = 0.8


def create_target(stock_df, horizon, l_threshold, h_threshold):
    """Create the target variable.

    This is always discretized into 3 bins.

    Parameters
    ----------
    stock_df : pd.DataFrame
        The DataFrame contains a Close column to evaluate target labels
    l_threshold : np.number
        The lower threshold as percentage number
    h_threshold : np.number
        The higher threshold as percentage number
    """

    c = stock_df["Close"]
    y = 100 * (-c.diff(-horizon)) / c  # 100 * (C_t+1 - C_t) / C_t

    # discretize price variations
    y = pd.cut(y, bins=[-np.inf, l_threshold, h_threshold, np.inf], labels=CLASSES)
    return y  # horizon final NaNs are introduced


def drop_initial_nans(data):
    """Remove initial rows from training, due to NaNs in features"""

    row_nans = data.isna().sum(axis=1)
    first_valid_idx = next(idx for idx, count in enumerate(row_nans) if count == 0)
    return data.iloc[first_valid_idx:], first_valid_idx


# def split_tabular_dataset(df, y):
#     """Split the input dataframe into training and testing sets.

#     We do not create a validation set since it is automatically handled by sklearn's
#     Grid Search utilities.
#     """

#     #  stock_df = df.drop(columns=["Open", "High", "Low", "Close", "Volume"])

#     X_train, X_test, y_train, y_test = train_test_split(
#         df, y, train_size=TRAIN_SIZE, shuffle=False
#     )

#     X_train = drop_initial_nans(X_train)
#     y_train = drop_initial_nans(y_train)
#     assert X_train.shape[0] == y_train.shape[0]

#     X_test = drop_final_nans(X_test)
#     y_test = drop_final_nans(y_test)
#     assert X_test.shape[0] == y_test.shape[0]

#     return X_train, X_test, y_train, y_test