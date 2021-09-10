from functools import reduce
from os.path import join, exists

from src.log import create_experiment
from joblib import Parallel, delayed, dump, load
import numpy as np
import pandas as pd
from itertools import product
from time import time
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from tqdm import tqdm
import logging
import random
import click
import random
import warnings

from src.models import create_dir, instantiate_classifier, FUZZY_CLASSIFIERS
from src.data import (
    get_sectors,
    load_OHLCV_files,
    create_target,
    load_stock_entities,
    USED_SECTORS,
)
from src.data.preparation import TRAIN_SIZE, drop_initial_nans
from src.models import lstm


logging.basicConfig(
    format="%(asctime)s:%(name)s:%(levelname)s:%(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


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


def oversample(X_train, y_train, method):
    from imblearn.over_sampling import ADASYN, SMOTE

    if method == "adasyn":
        return ADASYN().fit_resample(X_train, y_train)
    else:
        return SMOTE().fit_resample(X_train, y_train)


def get_stock_splits(
    stock_df,
    year,
    classifier,
    horizon,
    l_threshold,
    h_threshold,
    training_type,
    oversampling,
):

    # Create discrete targets
    targets = create_target(stock_df, horizon, l_threshold, h_threshold)
    assert len(targets) == len(stock_df)

    if classifier == "L3":
        stock_df = stock_df.drop(columns=["Open", "High", "Low", "Close", "Volume"])

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
        #  use the specified year as test
        # X_train = past_years
        # y_train = past_targets
        # X_test = stock_df.loc[year, :]
        # y_test = targets.loc[year]

    #  Drop initial days with nans due to the technical indicators
    X_train, first_valid_idx = drop_initial_nans(X_train)
    y_train = y_train.iloc[first_valid_idx:]

    # Drop last `horizon` days not to be predicted
    X_test = X_test.iloc[:-horizon]
    y_test = y_test.iloc[:-horizon]

    # if a stock is trained on a single target label, e.g. all days are HOLD,
    # just skip it, nothing can be learned
    if y_train.unique().size <= 1:
        raise RuntimeError()

    #  Oversampling for all but L3
    if oversampling and classifier != "L3":
        logger.info(f"Oversampling requested with {oversampling}")
        X_train, y_train = oversample(X_train, y_train, method=oversampling)

    return X_train, X_test, y_train, y_test


def test(model, X_test, y_test):
    return model.predict(X_test)


def train(
    X_train,
    y_train,
    tick: str,
    classifier: str,
    output_dir: str,
    do_grid_search: bool,
    normalize: bool,
    **classifier_kwargs,
):

    if do_grid_search:
        logger.info(f"Grid search for {classifier} requested.")

        clf, params, grid = instantiate_classifier(
            classifier, return_grid=True, **classifier_kwargs
        )

        # update param grid keys to match the use of pipeline
        if isinstance(grid, list):
            grid = [{f"clf__{k}": v for k, v in g.items()} for g in grid]
        else:
            grid = {f"clf__{k}": v for k, v in grid.items()}

        #  Normalize for all classifiers but L3
        if normalize:
            scaler = (
                MinMaxScaler() if classifier in FUZZY_CLASSIFIERS else StandardScaler()
            )
            pipeline = Pipeline([("scaler", scaler), ("clf", clf)])
        else:
            pipeline = Pipeline([("clf", clf)])

        n_jobs = 1 if classifier == "MLP" else -1
        gs = GridSearchCV(
            pipeline,
            param_grid=grid,
            scoring="f1_macro",
            n_jobs=n_jobs,
            cv=TimeSeriesSplit(n_splits=3),
            verbose=10,
        )

        if classifier == "L3":
            gs.fit(
                X_train,
                y_train,
                clf__column_names=X_train.columns,
                clf__remove_training_dir=True,
            )
        else:
            gs.fit(X_train, y_train)

        # save the best estimator
        dump(
            gs.best_estimator_, join(output_dir, "models", f"best_model_{tick}.joblib")
        )

        # for L3 save also the rules
        if classifier == "L3":
            create_dir(join(output_dir, "rules", tick))
            gs.best_estimator_.named_steps["clf"].save_rules(
                join(output_dir, "rules", tick)
            )

        return gs

    else:
        # TODO implement this if we ever need it
        raise RuntimeError("We don't want to do that now.")


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
@click.option("--oversampling", type=click.STRING, default=None)
@click.option("--h_threshold", type=click.FLOAT, default=1)
@click.option("--l_threshold", type=click.FLOAT, default=-1)
@click.option("--seed", type=click.INT, default=42)
@click.option("--test_run", is_flag=True)
@click.option("--parallel", is_flag=True)
@click.option("--n_workers", type=click.INT, default=16)
@click.option("--seq_length", type=click.INT, default=5)
@click.option("--batch_size", type=click.INT, default=128)
@click.option("--max_epochs", type=click.INT, default=30)
@click.option("--lr", type=click.FLOAT, default=2e-5)
@click.option("--early_stop", type=click.INT, default=0)
@click.option("--gpus", type=click.INT, default=1)
@click.option("--stateful", is_flag=True)
@click.option("--reduce_lr", type=click.INT, default=0)
@click.option("--rule_sets_modifier", type=click.STRING, default="standard")
@click.option("--use_sectors", is_flag=True)
def main(
    output_dir,
    classifier,
    year,
    training_type,
    horizon,
    log_comet,
    do_grid_search,
    normalize,
    oversampling,
    h_threshold,
    l_threshold,
    seed,
    test_run,
    parallel,
    n_workers,
    seq_length,
    batch_size,
    max_epochs,
    lr,
    early_stop,
    gpus,
    stateful,
    reduce_lr,
    rule_sets_modifier,
    use_sectors,
):

    hparams = locals()
    random.seed(seed)

    create_dir(output_dir)
    create_dir(join(output_dir, "models"))
    if classifier == "L3":
        create_dir(join(output_dir, "rules"))

    in_dir = (
        join("data", "processed", "SP500_technical")
        if classifier != "L3"
        else join("data", "processed", "SP500_technical_discretized", "DENSE")
    )
    stock_by_tick = load_OHLCV_files(in_dir)
    logger.info(f"Loaded {len(stock_by_tick)} stocks")

    #  sort ticks alphabetically
    ticks = sorted(list(stock_by_tick.keys()))

    if test_run:
        stock_by_tick = {
            "AAPL": stock_by_tick["AAPL"],
            "MSFT": stock_by_tick["MSFT"],
            "AMZN": stock_by_tick["AMZN"],
        }

    experiment = None
    if log_comet:
        exp = create_experiment()
        exp.add_tag("training")
        exp.log_parameters(hparams)
        exp.log_other("n_stocks", len(stock_by_tick))

    classifier_args = dict()
    if classifier == "L3":
        classifier_args["rule_sets_modifier"] = rule_sets_modifier
    if classifier == "MLP":
        classifier_args["random_state"] = seed
    if classifier == "LSTM":
        classifier_args["seed"] = seed

    if not use_sectors:
        """Run trading separately per stock."""

        results = list()

        for tick in tqdm(ticks, desc="Stocks"):
            stock_df = stock_by_tick[tick]

            X_train, X_test, y_train, y_test = get_stock_splits(
                stock_df,
                year,
                classifier,
                horizon,
                l_threshold,
                h_threshold,
                training_type,
                oversampling,
            )

            if classifier != "LSTM":

                model = train(
                    X_train,
                    y_train,
                    tick,
                    classifier,
                    output_dir,
                    do_grid_search,
                    normalize,
                    seed,
                    experiment,
                    **classifier_args,
                )

                y_pred = model.predict(X_test)
                best_params = model.best_params_

            else:

                #  LSTM
                logger.info("Disabling model and experiment logging with LSTM.")
                save_model = False
                comet_experiment = None

                best_model_path = lstm.train(
                    X_train,
                    y_train,
                    3,
                    seq_length,
                    batch_size,
                    max_epochs,
                    lr,
                    reduce_lr,
                    gpus,
                    seed,
                    early_stop,
                    stateful,
                    comet_experiment=comet_experiment,
                    model_dir=join(output_dir, "models"),
                    tick=tick,
                    save_model=save_model,
                )

                y_pred = lstm.test(
                    best_model_path,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    seq_length,
                    batch_size,
                )
                best_params = None
                os.remove(best_model_path)

            test_perf = score_classifier(y_test, y_pred)
            results.append((y_test, test_perf, y_pred, best_params))

    else:
        """Group stocks into sectors. We keep only the most populated ones."""

        if log_comet:
            exp.add_tag("sectors")

        logger.info(f"Training on {len(USED_SECTORS)} sectors")
        logger.info(f"Sectors: {USED_SECTORS}")

        stocks = load_stock_entities(join("data", "raw"))
        sectors = get_sectors(stocks)

        results = list()
        process_ticks = list()

        for sec in tqdm(USED_SECTORS, desc="Sectors"):

            curr_ticks = sectors.loc[sectors == sec].index.values
            process_ticks.extend(curr_ticks)

            Xtr, Xte, ytr, yte = list(), list(), list(), list()
            for tick in curr_ticks:
                stock_df = stock_by_tick[tick]
                X_train, X_test, y_train, y_test = get_stock_splits(
                    stock_df,
                    year,
                    classifier,
                    horizon,
                    l_threshold,
                    h_threshold,
                    training_type,
                    oversampling,
                )

                Xtr.append(X_train)
                Xte.append(X_test)
                ytr.append(y_train)
                yte.append(y_test)

            X_train = pd.concat(Xtr, axis=0)
            y_train = pd.concat(ytr, axis=0)

            logger.info(f"Dimensions of this sector: X_train: {X_train.shape}")

            if classifier != "LSTM":

                model = train(
                    X_train=X_train,
                    y_train=y_train,
                    tick=sec,
                    classifier=classifier,
                    output_dir=output_dir,
                    do_grid_search=do_grid_search,
                    normalize=normalize,
                    **classifier_args,
                )

                #  Predict one stock at a time within the current sector
                for tick, X_test, y_test in tqdm(
                    zip(curr_ticks, Xte, yte),
                    desc="Stocks",
                    leave=False,
                    total=len(curr_ticks),
                ):
                    y_pred = model.predict(X_test)
                    test_perf = score_classifier(y_test, y_pred)
                    results.append((y_test, test_perf, y_pred, model.best_params_))

            else:

                #  LSTM
                logger.info("Disabling model and experiment logging with LSTM.")
                save_model = False
                comet_experiment = None

                best_model_path = lstm.train(
                    X_train=X_train,
                    y_train=y_train,
                    num_classes=3,
                    seq_length=seq_length,
                    batch_size=batch_size,
                    max_epochs=max_epochs,
                    lr=lr,
                    reduce_lr=reduce_lr,
                    gpus=gpus,
                    seed=seed,
                    early_stop=early_stop,
                    stateful=stateful,
                    comet_experiment=comet_experiment,
                    model_dir=join(output_dir, "models"),
                    tick=sec,
                    save_model=save_model,
                )

                #  Predict one stock at a time within the current sector
                for tick, X_test, y_test in tqdm(
                    zip(curr_ticks, Xte, yte),
                    desc="Stocks",
                    leave=False,
                    total=len(curr_ticks),
                ):
                    y_pred = lstm.test(
                        best_model_path,
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        seq_length,
                        batch_size,
                    )
                    test_perf = score_classifier(y_test, y_pred)
                    results.append((y_test, test_perf, y_pred, None))

                os.remove(best_model_path)

        logger.info(f"Processed {len(results)} stocks")
        ticks = process_ticks

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
        best_params.to_csv(join(output_dir, f"best_params.csv"), index_label="tick")
        test_perf.to_csv(join(output_dir, "test_perf_GS.csv"), index_label="tick")
        test_pred.to_csv(
            join(output_dir, "preds", "test_preds_GS.csv"), index_label="Date"
        )

    else:
        test_perf.to_csv(join(output_dir, "test_perf.csv"), index_label="tick")
        test_pred.to_csv(
            join(output_dir, "preds", "test_preds.csv"), index_label="Date"
        )

    if log_comet:
        exp.log_metrics(test_perf.mean().to_dict())


if __name__ == "__main__":
    main()
