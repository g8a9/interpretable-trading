from os import listdir, mkdir
from os.path import join, exists

from comet_ml import experiment
from joblib import Parallel, delayed, dump, load
import numpy as np
import pandas as pd
from itertools import product
from time import time
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
import argparse
from tqdm import tqdm
from collections import namedtuple
import logging
import shutil
import glob

from src.models import create_dir
from src.log import create_experiment
from src.data import load_OHLCV_files

logger = logging.getLogger(__name__)

# def get_train_test(task, cvar_h, stock_tuple):
#     """Compute the training and testing sets for the current task and stock.

#     The only task now is have each year as a separate dataset.
#     """
#     stock_tick, stock_df = stock_tuple[0], stock_tuple[1]

#     # filter data of the current year. TODO Here different tasks
#     if "ONEYEAR" in task:
#         year = task.split("-")[1]
#         stock_df = stock_df.loc[year]
#     else:
#         raise NotImplementedError()

#     # get 'y' data
#     y = get_y(stock_df, cvar_h, L_THRESHOLD, H_THRESHOLD)

#     # split train test
#     X_train, X_test, y_train, y_test = train_test_split(stock_df, y)

#     # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
#     return stock_tick, X_train, X_test, y_train, y_test


def process_stock(
    task,  # the type of task
    disc,  # the type of discretization
    cvar_h,  # the time horizon for the current simulation
    out_dir,  #  the output folder
    out_dir_contents,
    args,
    stock_tick,
    X_train,
    X_test,
    y_train,
    y_test,
    classifier_file,
    timing_file,
):
    stime = time()

    #  if a stock was already processed, skip it
    if out_dir_contents is not None and f"pred_{stock_tick}.csv" in out_dir_contents:
        logger.info(f"{task, disc, cvar_h, stock_tick} already exists: skipping it.")
        return

    # if a stock is trained on a single target label, e.g. all days are HOLD,
    # just skip it, nothing can be learned from a single model
    if y_train.unique().size <= 1:
        logger.info(
            f"{task, disc, cvar_h}: skipping stock {stock_tick} due to single class in training set"
        )
        return

    if args.do_grid_search:
        #  perform grid search on parameters
        clf, param_grid = get_classifier_and_grid(args.classifier, return_grid=True)

        if args.classifier == Classifier.L3:
            pipeline = Pipeline([("clf", clf)])
        else:
            # introduce std normalization for classifiers on continuous variables
            pipeline = Pipeline([("scaler", StandardScaler()), ("clf", clf)])

        # update param grid keys to match the use of pipeline
        param_grid = {f"clf__{k}": v for k, v in param_grid.items()}
        gs = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            scoring="f1_weighted",
            n_jobs=-1,
            cv=TimeSeriesSplit(n_splits=5),
        )
        gs.fit(X_train, y_train)
        y_pred = gs.predict(
            X_test
        )  # use the best configuration found to predict the test set
        estimator = gs.best_estimator_  #  use the best estimator from now on

    else:
        # instantiate a model and validate it
        clf = get_classifier_from_config(args)
        if args.classifier == Classifier.L3:
            pipeline = Pipeline([("clf", clf)])
            pipeline.fit(
                X_train,
                y_train,
                clf__column_names=X_train.columns,
                clf__save_human_readable=args.save_human_readable,
            )
            if args.save_human_readable:  # save the rules into the output directory
                shutil.move(
                    pipeline["clf"].current_token_,
                    join(out_dir, f"{stock_tick}_{pipeline['clf'].current_token_}"),
                )
        else:
            # introduce std normalization for classifiers on continuous variables
            pipeline = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
            pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        estimator = pipeline["clf"]

    # save predictions and classifier's scores and timings
    # here we save either stats from the best gs estimator or the single estimator
    res_df = pd.DataFrame(
        [y_test.values, y_pred], columns=["y_true", "y_pred"], index=X_test.index
    )
    res_df.to_csv(join(out_dir, f"pred_{stock_tick}.csv"), header=True, index=True)

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1_all = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_updown = f1_score(
        y_test, y_pred, average="weighted", labels=[-1, 1], zero_division=0
    )

    with open(join(out_dir, classifier_file), "a") as fp:
        fp.write(f"{stock_tick},{estimator},{acc},{bal_acc},{f1_all},{f1_updown}\n")

    with open(join(out_dir, timing_file), "a") as fp:
        fp.write(f"{stock_tick},{int(time() - stime)}\n")

    # finally, save the model itself. If classifier=L3 here we save the rule sets and the
    # information on the transaction labeled
    dump(estimator, join(out_dir, f"estimator_{stock_tick}.joblib"), compress=3)


def main():
    parser = argparse.ArgumentParser(
        description="Use a SoTA ML classifier to predict price variations of stock prices."
    )
    parser.add_argument("job_name", type=str, help="The name of the output directory")
    parser.add_argument("classifier", type=str)
    parser.add_argument("year", type=int)
    parser.add_argument("sim_type", type=str, default="single_year")

    parser.add_argument("--log_comet", action="store_true")
    parser.add_argument("--do_grid_search", action="store_true")
    parser.add_argument("--save_human_readable", action="store_true")
    parser.add_argument(
        "--rule_sets_modifier", type=str, choices=["level1"], default=None
    )
    args = parser.parse_args()

    job_dir = args.job_name
    create_dir(job_dir)

    if args.log_comet:
        exp = create_experiment()
        exp.log_parameters(args)

    in_dir = (
        join("data", "processed", "SP500_technical_[2007,2017]")
        if args.classifier != "L3"
        else join("data", "processed", "technical_discretized_[2007,2017]", "DENSE")
    )
    stocks = load_OHLCV_files(in_dir)
    logger.info(f"Loaded {len(stocks)} stocks")

    # process a single configuration
    for task, disc, cvar_h in tqdm(configs, desc="Configs"):
        logging.info(f"Starting config: {task, disc, cvar_h}")
        # create a folder for a given configuration
        out_dir = join(job_dir, f"{task}_{args.classifier}_{disc}_{cvar_h}")
        out_dir_contents = None
        if exists(out_dir):
            out_dir_contents = listdir(out_dir)
        else:
            mkdir(out_dir)

        if disc == "COARSE":
            stocks_dict = stock_classic_dfs
        elif disc == "DENSE":
            stocks_dict = stock_express_dfs
        elif disc == "NO":
            stocks_dict = stock_dfs
        else:
            raise NotImplementedError()

        # Generate and store in memory each training and testing sets
        stime = time()
        config_stocks = product([task], [cvar_h], [disc], stocks_dict.items())
        data = Parallel(n_jobs=-1)(
            delayed(get_train_test)(c[0], c[1], c[3]) for c in config_stocks
        )
        with open(join(out_dir, timing_file), "a") as timing_fp:
            timing_fp.write(f"Preprocessing,{int(time() - stime)}\n")

        # process a single stock, given a configuration
        valtime = time()

        if args.classifier != Classifier.L3:
            for stock_tick, X_train, X_test, y_train, y_test in tqdm(
                data, desc="Stocks"
            ):
                process_stock(
                    task,
                    disc,
                    cvar_h,
                    out_dir,
                    out_dir_contents,
                    args,
                    stock_tick,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    classifier_file,
                    timing_file,
                )
        else:
            Parallel(n_jobs=-1)(
                delayed(process_stock)(
                    task,
                    disc,
                    cvar_h,
                    out_dir,
                    out_dir_contents,
                    args,
                    stock_tick,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    classifier_file,
                    timing_file,
                )
                for stock_tick, X_train, X_test, y_train, y_test in tqdm(
                    data, desc="Stocks"
                )
            )

        with open(join(out_dir, timing_file), "a") as timing_fp:
            timing_fp.write(f"TuningAndValidation,{int(time() - valtime)}\n")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()
