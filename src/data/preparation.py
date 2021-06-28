from l3wrapper.l3wrapper import TRAIN_BIN
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)


CLASSES = ["DOWN", "HOLD", "UP"]
TRAIN_SIZE = 0.8


def create_target(stock_df, cvar_h, l_threshold, h_threshold):
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
    y = 100 * (-c.diff(-cvar_h)) / c  # 100 * (C_t+1 - C_t) / C_t

    # discretize price variations
    y = pd.cut(y, bins=[-np.inf, l_threshold, h_threshold, np.inf], labels=CLASSES)
    return y  # cvar_h final NaNs are introduced


def drop_initial_nans(df):
    """Remove initial rows from training, due to NaNs in features"""
    first_valid_idx = next(ts for ts, c in df.isna().sum(axis=1).iteritems() if c == 0)
    logger.debug(f"First valid idx: {first_valid_idx}")
    return df.iloc[first_valid_idx:]


def drop_final_nans(df):
    # remove final rows from test, due to NaNs introduced by the horizon cvar_h
    first_invalid_idx = next(
        i for i, (ts, isna) in enumerate(df.isna().iteritems()) if isna
    )
    logging.debug(f"First invalid idx: {first_invalid_idx}")
    return df.iloc[:first_invalid_idx]


def split_tabular_dataset(df, y):
    """Split the input dataframe into training and testing sets.

    We do not create a validation set since it is automatically handled by sklearn's
    Grid Search utilities.
    """

    stock_df = df.drop(columns=["Open", "High", "Low", "Close", "Volume"])

    train_len = int(len(stock_df.index) * TRAIN_SIZE)
    logger.debug(f"train_len: {train_len}")

    X_train, X_test, y_train, y_test = train_test_split(
        stock_df, y, train_size=TRAIN_SIZE, shuffle=False
    )

    logger.debug(
        f"Initial train and test dimensions "
        f"{X_train.shape}, {y_train.shape}\n"
        f"{X_test.shape}, {y_test.shape}"
    )

    X_train = drop_initial_nans(X_train)
    y_train = drop_initial_nans(y_train)
    assert X_train.shape[0] == y_train.shape[0]

    X_test = drop_final_nans(X_test)
    y_test = drop_final_nans(y_test)
    assert X_test.shape[0] == y_test.shape[0]

    return X_train, X_test, y_train, y_test