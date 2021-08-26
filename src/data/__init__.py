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
            Stock(
                splitext(basename(f))[0], df.Open, df.Close, df.High, df.Low, df.Volume
            )
        )
    return stocks


def get_sectors(stocks) -> pd.Series:
    info = pd.read_csv(
        "./data/S&P500_wikipedia_sectors.csv", sep=";", index_col="Symbol"
    )
    sectors = info.reindex([s.name for s in stocks])["GICS Sector"]
    return sectors
