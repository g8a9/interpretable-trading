from os.path import join, exists


from joblib import Parallel, delayed, dump, load
import numpy as np
import pandas as pd
from itertools import product
from time import time
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import logging
import random
import click
import random

from src.models import (
    create_dir,
    instantiate_classifier,
)
from src.log import create_experiment
from src.data import load_OHLCV_files, create_target
from src.data.preparation import TRAIN_SIZE, drop_initial_nans


logging.basicConfig(
    format="%(asctime)s:%(name)s:%(levelname)s:%(message)s",
    level=logging.INFO,
)
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


def score_classifier(y_true, y_pred) -> dict:
    """Compute performance measures."""

    perf = {
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_pos": f1_score(
            y_true, y_pred, labels=[0, 2], average="weighted", zero_division=0
        ),
    }
    return perf


def process_stock(
    tick: str,
    stock_df: pd.DataFrame,
    classifier: str,
    output_dir: str,
    training_type: str,
    year: str,
    horizon: int,
    h_threshold: float,
    l_threshold: float,
    do_grid_search: bool,
    normalize: bool,
    oversample: bool,
    experiment,
):
    """Process a single stock.

    Steps: TODO
    """
    config = (tick, year, horizon, training_type)

    stime = time()

    # scores_file = get_scores_filename(tick)
    # scores_path = join(output_dir, scores_file)
    # if exists(scores_path):
    #     logger.info(f"Scores for config {config}  already exists: skipping it.")

    # Create discrete targets
    targets = create_target(stock_df, horizon, l_threshold, h_threshold)
    assert len(targets) == len(stock_df)

    #  Create training and testing split (eventually validation)
    X, y = stock_df.loc[year, :], targets.loc[year]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=TRAIN_SIZE, shuffle=False
    )

    if training_type == "cumulative":
        past_years = stock_df.loc[: str(int(year) - 1), :]
        past_targets = targets.loc[: str(int(year) - 1)]
        X_train = pd.concat([past_years, X_train], axis=0)
        y_train = pd.concat([past_targets, y_train], axis=0)

    #  Drop initial days with nans due to the technical indicators
    X_train, first_valid_idx = drop_initial_nans(X_train)
    y_train = y_train.iloc[first_valid_idx:]

    # Drop last `horizon` days not to be predicted
    X_test = X_test.iloc[:-horizon]
    y_test = y_test.iloc[:-horizon]

    # if a stock is trained on a single target label, e.g. all days are HOLD,
    # just skip it, nothing can be learned
    if y_train.unique().size <= 1:
        logger.info(f"Skipping {config} due to a single class in the training set")
        return

    if do_grid_search:
        clf, params, grid = instantiate_classifier(classifier, return_grid=True)

        # update param grid keys to match the use of pipeline
        grid = {f"clf__{k}": v for k, v in grid.items()}

        if normalize:
            pipeline = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        else:
            pipeline = Pipeline([("clf", clf)])

        gs = GridSearchCV(
            pipeline,
            param_grid=grid,
            scoring="f1_macro",
            n_jobs=-1,
            cv=TimeSeriesSplit(n_splits=5),
        )
        gs.fit(X_train, y_train)

        test_pred = gs.predict(X_test)
        test_performance = score_classifier(y_test, test_pred)
        return y_test, test_performance, test_pred, gs.best_params_

    else:
        # instantiate a model and validate it
        clf, _ = instantiate_classifier(classifier)

        if normalize:
            pipeline = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        else:
            pipeline = Pipeline([("clf", clf)])

        pipeline.fit(X_train, y_train)
        test_pred = pipeline.predict(X_test)
        test_performance = score_classifier(y_test, test_pred)

        return y_test, test_performance, test_pred

    #     if args.classifier == Classifier.L3:
    #         pipeline = Pipeline([("clf", clf)])
    #         pipeline.fit(
    #             X_train,
    #             y_train,
    #             clf__column_names=X_train.columns,
    #             clf__save_human_readable=args.save_human_readable,
    #         )
    #         if args.save_human_readable:  # save the rules into the output directory
    #             shutil.move(
    #                 pipeline["clf"].current_token_,
    #                 join(out_dir, f"{stock_tick}_{pipeline['clf'].current_token_}"),
    #             )
    #     else:
    #         # introduce std normalization for classifiers on continuous variables
    #         pipeline = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    #         pipeline.fit(X_train, y_train)

    #     y_pred = pipeline.predict(X_test)
    #     estimator = pipeline["clf"]

    # # save predictions and classifier's scores and timings
    # # here we save either stats from the best gs estimator or the single estimator
    # res_df = pd.DataFrame(
    #     [y_test.values, y_pred], columns=["y_true", "y_pred"], index=X_test.index
    # )
    # res_df.to_csv(join(out_dir, f"pred_{stock_tick}.csv"), header=True, index=True)

    # with open(join(out_dir, classifier_file), "a") as fp:
    #     fp.write(f"{stock_tick},{estimator},{acc},{bal_acc},{f1_all},{f1_updown}\n")

    # with open(join(out_dir, timing_file), "a") as fp:
    #     fp.write(f"{stock_tick},{int(time() - stime)}\n")

    # # finally, save the model itself. If classifier=L3 here we save the rule sets and the
    # # information on the transaction labeled
    # dump(estimator, join(out_dir, f"estimator_{stock_tick}.joblib"), compress=3)


@click.command()
@click.argument("output_dir", type=click.Path(file_okay=False, writable=True))
@click.argument("classifier", type=click.STRING)
@click.argument("year", type=click.STRING)
@click.option(
    "--training_type", type=click.Choice(["year", "cumulative"]), default="year"
)
@click.option("--horizon", type=click.INT, default=1)
@click.option("--log_comet", is_flag=True)
@click.option("--do_grid_search", is_flag=True)
@click.option("--normalize", is_flag=True)
@click.option("--oversample", type=click.STRING, default=None)
@click.option("--h_threshold", type=click.FLOAT, default=1)
@click.option("--l_threshold", type=click.FLOAT, default=-1)
@click.option("--seed", type=click.INT, default=42)
@click.option("--save_human_readable", is_flag=True)
@click.option("--test_run", is_flag=True)
def main(
    output_dir,
    classifier,
    year,
    training_type,
    horizon,
    log_comet,
    do_grid_search,
    normalize,
    oversample,
    h_threshold,
    l_threshold,
    seed,
    save_human_readable,
    test_run,
):
    hparams = locals()
    random.seed(seed)

    # parser.add_argument(
    #     "--rule_sets_modifier", type=str, choices=["level1"], default=None
    # )
    # parser.add_argument("--test_run", action="store_true")
    # args = parser.parse_args()

    create_dir(output_dir)

    in_dir = (
        join("data", "processed", "SP500_technical")
        if classifier != "L3"
        else join("data", "processed", "technical_discretized", "DENSE")
    )
    stocks = load_OHLCV_files(in_dir)
    ticks, stock_dfs = zip(*stocks)
    logger.info(f"Loaded {len(stocks)} stocks")

    if test_run:
        stocks = random.sample(stocks, k=3)

    experiment = None
    if log_comet:
        exp = create_experiment()
        exp.add_tag("training")
        exp.log_parameters(hparams)
        exp.log_other("n_stocks", len(stocks))

    results = Parallel(n_jobs=1)(
        delayed(process_stock)(
            tick,
            stock_df,
            classifier,
            output_dir,
            training_type,
            year,
            horizon,
            h_threshold,
            l_threshold,
            do_grid_search,
            normalize,
            oversample,
            experiment,
        )
        for tick, stock_df in tqdm(stocks, desc="Stocks")
    )

    # Save all the results
    create_dir(join(output_dir, "preds"))

    y_tests = [r[0] for r in results]
    y_tests = pd.concat(y_tests, axis=1, keys=ticks)
    y_tests.to_csv(join(output_dir, "preds", "test_gold.csv"), index_label="Date")

    test_perf = [r[1] for r in results]
    test_perf = pd.DataFrame(test_perf, index=ticks)

    test_pred = [r[2] for r in results]
    test_pred = np.array(test_pred).transpose()
    test_pred = pd.DataFrame(test_pred, index=y_tests.index, columns=ticks)

    if do_grid_search:
        best_params = [r[3] for r in results]
        best_params = pd.DataFrame(best_params, index=ticks)
        best_params.to_csv(join(output_dir, "best_params.csv"), index_label="tick")
        test_perf.to_csv(join(output_dir, "test_perf_GS.csv"), index_label="tick")
        test_pred.to_csv(
            join(output_dir, "preds", "test_preds_GS.csv"), index_label="Date"
        )

    else:
        test_perf.to_csv(join(output_dir, "test_perf.csv"), index_label="tick")
        test_pred.to_csv(
            join(output_dir, "preds", "test_preds.csv"), index_label="Date"
        )

    return

    # # process a single configuration
    # for task, disc, cvar_h in tqdm(configs, desc="Configs"):
    #     logging.info(f"Starting config: {task, disc, cvar_h}")
    #     # create a folder for a given configuration
    #     out_dir = join(job_dir, f"{task}_{args.classifier}_{disc}_{cvar_h}")
    #     out_dir_contents = None
    #     if exists(out_dir):
    #         out_dir_contents = listdir(out_dir)
    #     else:
    #         mkdir(out_dir)

    #     if disc == "COARSE":
    #         stocks_dict = stock_classic_dfs
    #     elif disc == "DENSE":
    #         stocks_dict = stock_express_dfs
    #     elif disc == "NO":
    #         stocks_dict = stock_dfs
    #     else:
    #         raise NotImplementedError()

    #     # Generate and store in memory each training and testing sets
    #     stime = time()
    #     config_stocks = product([task], [cvar_h], [disc], stocks_dict.items())
    #     data = Parallel(n_jobs=-1)(
    #         delayed(get_train_test)(c[0], c[1], c[3]) for c in config_stocks
    #     )
    #     with open(join(out_dir, timing_file), "a") as timing_fp:
    #         timing_fp.write(f"Preprocessing,{int(time() - stime)}\n")

    #     # process a single stock, given a configuration
    #     valtime = time()

    #     if args.classifier != Classifier.L3:
    #         for stock_tick, X_train, X_test, y_train, y_test in tqdm(
    #             data, desc="Stocks"
    #         ):
    #             process_stock(
    #                 task,
    #                 disc,
    #                 cvar_h,
    #                 out_dir,
    #                 out_dir_contents,
    #                 args,
    #                 stock_tick,
    #                 X_train,
    #                 X_test,
    #                 y_train,
    #                 y_test,
    #                 classifier_file,
    #                 timing_file,
    #             )
    #     else:
    #         Parallel(n_jobs=-1)(
    #             delayed(process_stock)(
    #                 task,
    #                 disc,
    #                 cvar_h,
    #                 out_dir,
    #                 out_dir_contents,
    #                 args,
    #                 stock_tick,
    #                 X_train,
    #                 X_test,
    #                 y_train,
    #                 y_test,
    #                 classifier_file,
    #                 timing_file,
    #             )
    #             for stock_tick, X_train, X_test, y_train, y_test in tqdm(
    #                 data, desc="Stocks"
    #             )
    #         )

    #     with open(join(out_dir, timing_file), "a") as timing_fp:
    #         timing_fp.write(f"TuningAndValidation,{int(time() - valtime)}\n")


if __name__ == "__main__":
    main()
