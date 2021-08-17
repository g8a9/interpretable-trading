from collections import namedtuple
from glob import glob
import pandas as pd
from os.path import basename, splitext
from .make_dataset import load_OHLCV_files
from .preparation import create_target

Stock = namedtuple("Stock", ["name", "open", "close", "high", "low", "volume"])


def load_stock_entities(directory: str):
    files = glob(f"{directory}/*.csv")
    stocks = list()
    for f in files:
        df = pd.read_csv(f, index_col=0, header=0, parse_dates=True)
        stocks.append(
            Stock(splitext(basename(f))[0], df.Open, df.Close, df.High, df.Low, df.Volume)
        )
    return stocks