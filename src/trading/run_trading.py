#!/usr/bin/env python
# coding: utf-8

from os import listdir, mkdir
from os.path import join, exists

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
    DEFAULT_COLOR,
    DEFAULT_MARKER,
    DEFAULT_LINE_WIDTH,
    DEFAULT_MARKER_SIZE,
)

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", context="paper", font_scale=2.3, rc={"lines.linewidth": 2.3})


def log_fig(experiment, file_path):
    if not experiment:
        experiment = create_experiment()

    experiment.log_image(file_path)


def main():
    parser = ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("year", type=str)

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

    #  Setup output files and folders
    CLASSIFIER_COMPARISON_FILE = "classifiers_comparison.csv"
    CLASSIFIER_SUMMARY_FILE = "classifiers_summary.csv"
    YEARLY_MEAN_FILE = "yearly_mean.csv"
    EQUITY_FIG_FILE = "equity.pdf"
    OP_FILE = "operations.csv"
    OP_TREND_FILE = "operations_trend.pdf"
    OP_VS_EQUITY_FILE = "operations_vs_equity.pdf"
    if not exists(output_dir):
        mkdir(output_dir)

    summary_stats = list()
    trading_stats = list()
    equity_fig, equity_ax = plt.subplots(figsize=(14, 8))

    for classifier in tqdm(
        ["L3", "L3-LVL1", "GNB", "SVC", "KNN", "LG", "RFC", "MLP"], desc="Clf"
    ):
        sig_file = join(
            args.input_dir,
            f"out_{args.year}_{classifier}",
            "preds",
            "test_preds_GS.csv",
        )
        if not exists(sig_file):
            print("Can't find file with trading signals for", classifier)
            continue
        else:
            signals_df = pd.read_csv(
                sig_file,
                parse_dates=["Date"],
                header=0,
                infer_datetime_format=True,
                index_col="Date",
            )

        assert not signals_df.isna().any().any()
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
        assert len(positions) > 0, "At least on position should have been opened"

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
        trading_stats.append(ts)
        summary_stats.append(ts)

        #  save operations history
        positions_df.to_csv(
            join(output_dir, f"{classifier}_{OP_FILE}"),
            index=False,
            float_format="%.3f",
        )

        # equity vs other classifiers
        label = EQUITY_LINES[classifier].get("label", classifier)
        color = EQUITY_LINES[classifier].get("color", DEFAULT_COLOR)
        marker = EQUITY_LINES[classifier].get("marker", DEFAULT_MARKER)
        marker_size = EQUITY_LINES[classifier].get("marker_size", DEFAULT_MARKER_SIZE)
        lw = EQUITY_LINES[classifier].get("lw", DEFAULT_LINE_WIDTH)

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

        # operations opened vs equity
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

        # operations trend
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

    if args.log_comet:
        experiment.log_figure(figure=equity_fig, figure_name=EQUITY_FIG_FILE)

    plt.close(equity_fig)

    # if args.log_comet:
    #     experiment.log_image(join(output_dir, EQUITY_FIG_FILE))

    trading_stats_df = pd.DataFrame(trading_stats)
    trading_stats_df.to_csv(
        join(output_dir, CLASSIFIER_COMPARISON_FILE),
        index=False,
        float_format="%.3f",
    )

    if args.log_comet:
        experiment.log_table(CLASSIFIER_COMPARISON_FILE, trading_stats_df)

    summary_df = pd.DataFrame(summary_stats)
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