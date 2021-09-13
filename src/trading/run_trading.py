#!/usr/bin/env python
# coding: utf-8

from os import listdir, makedirs
from os.path import join, exists

from seaborn.rcmod import reset_defaults

from src.log import create_experiment

from tqdm import tqdm
import pandas as pd
import numpy as np
from collections import namedtuple
import logging
from argparse import ArgumentParser

from src.trading import trading
from src.data import load_stock_entities
from src.trading.style import (
    EQUITY_LINES,
    EQUITY_DEFAULT,
)
from src.models import FUZZY_CLASSIFIERS, DEEPRL_SYSTEMS


logging.basicConfig(
    format="%(asctime)s:%(name)s:%(levelname)s:%(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", context="paper", font_scale=2.3, rc={"lines.linewidth": 2.3})

#  Setup output files and folders
CLASSIFIER_COMPARISON_FILE = "classifiers_comparison.csv"
CLASSIFIER_SUMMARY_FILE = "classifiers_summary.csv"
YEARLY_MEAN_FILE = "yearly_mean.csv"
EQUITY_FILE = "equity.csv"
EQUITY_FIG_FILE = "equity.pdf"
OP_FILE = "operations.csv"
OP_TREND_FILE = "operations_trend.pdf"
OP_VS_EQUITY_FILE = "operations_vs_equity.pdf"

TradingStats = namedtuple(
    "TradingStats",
    [
        "Year",
        "Classifier",
        "EquityFinal",
        "EquityMax",
        "EquityMin",
        "EquityMean",
        "EquityStd",
        "OpCount",
        "OpLONGCount",
        "OpSHORTCount",
        "OpClosedSTOPLOSS_perc",
        "OpClosedINVERSE_perc",
        "OpClosedMAXLENGTH_perc",
        "OpLengthMean_days",
        "OpLengthStd_days",
        "OpReturnMean",
        "OpReturnStd",
        "OpProfitMean_perc",
        "OpProfitStd_perc",
        "OpCapInvestedMean",
        "OpCapInvestedStd",
        "OpFeesMean",
        "OpFeesStd",
        "OpPerStockMean",
        "OpPerStockStd",
    ],
)


def log_fig(experiment, file_path):
    if not experiment:
        experiment = create_experiment()

    experiment.log_image(file_path)


def trade_with_signals(args, classifier, stocks, signals_df, output_dir):

    simulation = trading.TradingSimulation(
        initial_capital=args.initial_capital,
        stop_loss_tr=args.stop_loss_tr,
        investment=args.investment,
        stocks=stocks,
        fixed_fee=args.fixed_fee,
        percentage_fee=args.percentage_fee,
        max_opened=args.max_opened,
        max_opened_per_day=args.max_opened_per_day,
        max_duration=args.max_duration,
    )
    logging.debug(
        f"Trading simulation setup, "
        f"Initial capital {args.initial_capital}, "
        f"stop loss tr {args.stop_loss_tr}, "
        f"investment per day {args.investment}, "
        f"num stocks {len(stocks)}, "
        f"fixed fee {args.fixed_fee}, "
        f"percentage_fee {args.percentage_fee}, "
        f"max opened {args.max_opened}, "
        f"max opened per day {args.max_opened_per_day}, "
        f"max_duration {args.max_duration}."
    )
    positions, results_df = simulation.trade(signals_df)
    positions_df = pd.DataFrame([p.to_dict() for p in positions])

    if not positions:
        logger.warn(f"No positions opened for {classifier}, {args.year}")
        ts = TradingStats(
            args.year,
            classifier,
            args.initial_capital,
            args.initial_capital,
            args.initial_capital,
            args.initial_capital,
            args.initial_capital,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0
        )
    else:

        # create row for classifiers comparison
        ts = TradingStats(
            args.year,
            classifier,
            results_df.equity_by_day.iloc[-1],
            results_df.equity_by_day.max(),
            results_df.equity_by_day.min(),
            results_df.equity_by_day.mean(),
            results_df.equity_by_day.std(),
            positions_df.shape[0],
            positions_df.loc[
                positions_df.optype == trading.TradingOperationType.LONG
            ].shape[0],
            positions_df.loc[
                positions_df.optype == trading.TradingOperationType.SHORT
            ].shape[0],
            100
            * positions_df.loc[
                positions_df.close_cause == str(trading.CloseCause.STOP_LOSS)
            ].shape[0]
            / positions_df.shape[0],
            100
            * positions_df.loc[
                positions_df.close_cause == str(trading.CloseCause.INVERSE)
            ].shape[0]
            / positions_df.shape[0],
            100
            * positions_df.loc[
                positions_df.close_cause == str(trading.CloseCause.MAX_LENGTH)
            ].shape[0]
            / positions_df.shape[0],
            positions_df.length_in_days.mean(),
            positions_df.length_in_days.std(),
            positions_df["return"].mean(),
            positions_df["return"].std(),
            positions_df.profit_perc.mean(),
            positions_df.profit_perc.std(),
            positions_df.cap_invested.mean(),
            positions_df.cap_invested.std(),
            positions_df.fees.mean(),
            positions_df.fees.std(),
            positions_df.groupby("stock").open_day.count().mean(),
            positions_df.groupby("stock").open_day.count().std(),
        )


    #  save operations history
    positions_df.to_csv(
        join(output_dir, f"{classifier}_{OP_FILE}"),
        index=False,
        float_format="%.3f",
    )

    return positions, results_df, ts


def read_signals(args, classifier):
    sig_file = join(
        args.input_dir,
        f"out_{args.year}_{classifier}",
        "preds",
        "test_preds_GS.csv",
    )

    if not exists(sig_file):
        sig_file = join(
            args.input_dir,
            f"out_{args.year}_{classifier}",
            "preds",
            "test_preds.csv",
        )

    if not exists(sig_file):
        logger.info(f"Can't find file with trading signals for {classifier}")
        return

    else:
        signals_df = pd.read_csv(
            sig_file,
            parse_dates=["Date"],
            header=0,
            infer_datetime_format=True,
            index_col="Date",
        )

    assert not signals_df.isna().any().any()

    return signals_df


def get_classifiers(family="ml"):
    determ = ["L3", "L3-LVL1", "GNB", "SVC", "KNN", "LG", "RFC"]
    seeded = [f"MLP_{i}" for i in range(10)] + [f"LSTM_{i}" for i in range(10)]

    if family == "ml":
        return determ + seeded

    elif family == "fuzzy":
        return ["L3", "L3-LVL1"] + FUZZY_CLASSIFIERS

    elif family == "ml+fuzzy":
        return determ + seeded + FUZZY_CLASSIFIERS

    elif family == "deeprl":
        return ["L3", "L3-LVL1"] + DEEPRL_SYSTEMS

    else:
        raise ValueError()


def main():
    parser = ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--year", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=".")

    parser.add_argument("--stop_loss_tr", type=float, default=0.01)
    parser.add_argument(
        "--investment",
        type=float,
        help="Amount of capital invested each day",
        default=1,
    )
    parser.add_argument("--max_opened", type=int, default=9)
    parser.add_argument("--max_opened_per_day", type=int, default=3)
    parser.add_argument("--max_duration", type=int, default=3)
    parser.add_argument("--initial_capital", type=float, default=100000)
    parser.add_argument("--fixed_fee", type=float, default=0.0)
    parser.add_argument("--percentage_fee", type=float, default=0.0015)

    parser.add_argument("--log_comet", action="store_true")
    parser.add_argument("--family", type=str, default="ml")
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    if args.log_comet:
        experiment = create_experiment()
        experiment.add_tag("trading")
        experiment.log_parameters(args)

    output_dir = (
        f"rt_I{args.investment}_"
        f"SL{args.stop_loss_tr}_"
        f"MO{args.max_opened}_"
        f"MOPD{args.max_opened_per_day}_"
        f"MD{args.max_duration}_"
        f"{args.year}"
    )
    output_dir = join(args.output_dir, output_dir)

    # list of named tuples
    stocks = load_stock_entities("data/raw")

    if not exists(output_dir):
        makedirs(output_dir, exist_ok=True)

    summary_stats = list()
    equity_fig, equity_ax = plt.subplots(figsize=(14, 8))

    seeded_results = list()
    equities = list()
    for classifier in tqdm(get_classifiers(args.family), desc="Clf"):

        if classifier in DEEPRL_SYSTEMS:
            equity_file = join(args.input_dir, f"{classifier}_{args.year}_account_value.csv")
            if not exists(equity_file):
                logger.info(f"Signals are empty, skipping {classifier}")
                continue

            results_df = pd.read_csv(
                equity_file,
                parse_dates=["date"],
                header=0,
                infer_datetime_format=True,
                index_col="date",
                usecols=["date", "account_value"]
            ).rename(columns={"account_value": "equity_by_day"})

        else:
            signals_df = read_signals(args, classifier)
            if signals_df is None:
                logger.info(f"Signals are empty, skipping {classifier}")
                continue

            #  TRADING
            pos, results_df, ts = trade_with_signals(
                args, classifier, stocks, signals_df, output_dir
            )
            summary_stats.append(ts)

        equities.append(results_df.equity_by_day)

        #  VISUALIZATION

        base_classifier = classifier.split("_")[0]

        if base_classifier in EQUITY_LINES:
            # equity vs other classifiers
            label = EQUITY_LINES[base_classifier].get("label", base_classifier)
            color = EQUITY_LINES[base_classifier].get("color", EQUITY_DEFAULT["COLOR"])
            marker = EQUITY_LINES[base_classifier].get("marker", EQUITY_DEFAULT["MARKER"])
            marker_size = EQUITY_LINES[base_classifier].get(
                "marker_size", EQUITY_DEFAULT["MARKER_SIZE"]
            )
            lw = EQUITY_LINES[base_classifier].get("lw", EQUITY_DEFAULT["LINE_WIDTH"])
        else:
            label = base_classifier
            color = EQUITY_DEFAULT["COLOR"]
            marker = EQUITY_DEFAULT["MARKER"]
            marker_size = EQUITY_DEFAULT["MARKER_SIZE"]
            lw = EQUITY_DEFAULT["LINE_WIDTH"]

        # - Plot: equity line

        # Display differently models with seeds
        if classifier.startswith("MLP") or classifier.startswith("LSTM"):

            seeded_results.append(results_df)

            if len(seeded_results) == 10:
                #  I have collected 10 results, one per seed
                logger.info("Collecting results from different seeds")
                results_df = pd.concat(seeded_results, axis=0)
                sns.lineplot(
                    x=results_df.index,
                    y=results_df.equity_by_day,
                    ax=equity_ax,
                    label=label,
                    markersize=marker_size,
                    marker=marker,
                    color=color,
                    lw=lw,
                )
                seeded_results = list()

        else:
            sns.lineplot(
                x=results_df.index,
                y=results_df.equity_by_day,
                ax=equity_ax,
                label=label,
                markersize=marker_size,
                marker=marker,
                color=color,
                lw=lw,
            )


        # don't generate other charts for DEEPR SYSTEMS 
        if classifier in DEEPRL_SYSTEMS:
            continue

        # - Plot: operations opened vs equity

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(results_df.index, results_df.equity_by_day, label="equity")
        ax.set_ylabel("equity")
        ax2 = ax.twinx()
        ax2.set_ylabel("position opened")
        ax2.plot(
            results_df.index,
            results_df.total_opened,
            label="position opened",
            color="r",
        )
        fig.tight_layout()
        fig.legend()
        fig.savefig(join(output_dir, f"{classifier}_{OP_VS_EQUITY_FILE}"))

        if args.log_comet:
            experiment.log_figure(
                figure=fig, figure_name=f"{classifier}_{OP_VS_EQUITY_FILE}"
            )

        plt.close(fig)

        # if args.log_comet:
        #     experiment.log_image(join(output_dir, f"{classifier}_{OP_VS_EQUITY_FILE}"))

        # - Plot: operations trend

        clf_fig, clf_ax = plt.subplots(figsize=(12, 8))
        clf_ax.plot(results_df.index, results_df.total_opened, label="position opened")
        # clf_ax2 = clf_ax.twinx()
        clf_ax.bar(
            results_df.index,
            results_df.opened_by_day,
            label="opened by day",
            color="g",
        )
        clf_ax.bar(
            results_df.index,
            results_df.closed_by_day,
            label="closed by day",
            color="black",
            alpha=0.4,
        )
        clf_fig.legend()
        clf_fig.tight_layout()
        clf_fig.savefig(join(output_dir, f"{classifier}_{OP_TREND_FILE}"))

        if args.log_comet:
            experiment.log_figure(
                figure=clf_fig, figure_name=f"{classifier}_{OP_TREND_FILE}"
            )

        plt.close(clf_fig)

        # if args.log_comet:
        #     experiment.log_image(join(output_dir, f"{classifier}_{OP_TREND_FILE}"))


    ticks = pd.to_datetime(
        np.linspace(results_df.index[0].value, results_df.index[-1].value, 6)
    )
    equity_fig.tight_layout()
    equity_ax.legend(loc="upper left")
    equity_ax.set_ylabel("Equity")
    equity_ax.set_xlabel("Date")
    equity_ax.set_xticks(ticks)
    equity_fig.savefig(join(output_dir, EQUITY_FIG_FILE))

    # save a file with equities
    equities = pd.concat(equities, axis=1, keys=get_classifiers(args.family))
    equities.to_csv(join(output_dir, EQUITY_FILE), index_label="Date")

    if args.log_comet:
        experiment.log_figure(figure=equity_fig, figure_name=EQUITY_FIG_FILE)

    plt.close(equity_fig)

    summary_df = pd.DataFrame(summary_stats)
    if args.log_comet:
        experiment.log_table(CLASSIFIER_COMPARISON_FILE, summary_df)

    clfs_df = summary_df.groupby("Classifier")
    clfs_equity_mean = clfs_df["EquityFinal"].mean()
    clfs_equity_std = clfs_df["EquityFinal"].std()
    clfs_equity_mean.name = "EquityMean"
    clfs_equity_std.name = "EquityStd"
    pd.concat([clfs_equity_mean, clfs_equity_std], axis=1).to_csv(
        join(output_dir, YEARLY_MEAN_FILE), float_format="%.3f"
    )
    summary_df.to_csv(
        join(output_dir, CLASSIFIER_SUMMARY_FILE), index=False, float_format="%.3f"
    )


if __name__ == "__main__":
    main()
