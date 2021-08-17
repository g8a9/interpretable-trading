# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import glob
from os.path import join, basename, splitext

from dotenv import find_dotenv, load_dotenv
import pandas as pd
import ta
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def load_OHLCV_files(in_dir):
    files = glob.glob(join(in_dir, "*.csv"))

    ticks_stocks = list()
    for f in files:
        tick = splitext(basename(f))[0]
        ticks_stocks.append(
            (tick, pd.read_csv(f, index_col="Date", header=0, parse_dates=["Date"]))
        )
    return ticks_stocks


def extract_ta_features(stock_df):
    """Compute features using TA's technical indicators."""
    initial_len = len(stock_df)

    stock_df = ta.add_trend_ta(stock_df, high="High", low="Low", close="Close")
    stock_df = ta.add_momentum_ta(
        stock_df, high="High", low="Low", close="Close", volume="Volume"
    )
    stock_df = ta.add_volatility_ta(stock_df, high="High", low="Low", close="Close")
    stock_df = ta.add_volume_ta(
        stock_df, high="High", low="Low", close="Close", volume="Volume"
    )
    assert initial_len == len(stock_df)

    return stock_df


N_WORKERS = 16


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
@click.argument("output_version", type=click.STRING)
def main(input_filepath, output_filepath, output_version):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    stocks = load_OHLCV_files(input_filepath)

    if output_version == "v2":

        logger.info(f"Start processing {len(stocks)} dataframes...")
        with ThreadPoolExecutor(N_WORKERS) as p:
            stocks_processed = list(
                tqdm(p.map(extract_ta_features, stocks.values()), total=len(stocks))
            )

        assert len(stocks) == len(stocks_processed)

        for tick, stock_df in zip(stocks.keys(), stocks_processed):
            stock_df.to_csv(join(output_filepath, f"{tick}.csv"), index=False)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
