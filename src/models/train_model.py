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
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm
import logging
import random
import click
import random

from src.models import (
    create_dir,
    instantiate_classifier,
)
from src.data import load_OHLCV_files, create_target
from src.data.preparation import TRAIN_SIZE, drop_initial_nans


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


def evaluate_classifier(classifier, X_train, X_val, y_train, y_val):
    """Objective function"""

    def _evaluate(parameters):
        clf = SVC(**parameters)
        clf.fit(X_train, y_train)
        return {"f1_macro": (f1_score(y_val, clf.predict(X_val), average="macro"), 0.0)}

    if classifier == "SVC":
        # kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf"])
        # degree = trial.suggest_int("degree", 2, 4)
        # C = trial.suggest_loguniform("C", 1e-3, 10)

        # clf = SVC(kernel=kernel, degree=degree, C=C)
        # clf.fit(X_train, y_train)

        # y_pred = clf.predict(X_val)
        # score = f1_score(y_val, y_pred, average="macro", labels=[0, 2])
        # return score

        parameters = [
            {"name": "kernel", "type": "string", "values": ["linear", "poly", "rbf"]},
            {"name": "degree", "type": "range", "bounds": [3, 4], "value_type": "int"},
            {
                "name": "C",
                "type": "range",
                "bounds": [1e-4, 10],
                "value_type": "float",
                "log_scale": True,
            },
        ]
        best_parameters, values, experiment, model = optimize(
            parameters=parameters,
            evaluation_function=_evaluate,
            objective_name="f1_macro",
            experiment_name="test",
            minimize=False,
            total_trials=4,
        )
        print(best_parameters)


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
    oversampling: bool,
    seed: int,
    experiment,
):
    """Process a single stock."""
    config = (tick, year, horizon, training_type)

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

    if oversampling:
        X_train_, y_train_ = oversample(X_train, y_train, method=oversampling)

    if do_grid_search:
        # X_train_, X_val_, y_train_, y_val_ = train_test_split(
        #     X_train, y_train, train_size=0.8, shuffle=False
        # )

        # study = optuna.create_study(
        #     study_name=f"study_{tick}_{year}", direction="maximize"
        # )
        # study.optimize(
        #     lambda t: evaluate_classifier(
        #         t, classifier, X_train_, X_val_, y_train_, y_val_
        #     ),
        #     n_trials=20,
        #     timeout=120,
        #     n_jobs=-1,
        # )
        # print(study.best_params)

        # evaluate_classifier(classifier, X_train_, X_val_, y_train_, y_val_)

        clf, params, grid = instantiate_classifier(classifier, return_grid=True)

        # update param grid keys to match the use of pipeline
        if isinstance(grid, list):
            grid = [{f"clf__{k}": v for k, v in g.items()} for g in grid]
        else:
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
            cv=TimeSeriesSplit(n_splits=3),
        )
        gs.fit(X_train, y_train)

        test_pred = gs.predict(X_test)
        test_performance = score_classifier(y_test, test_pred)

        # save the best estimator
        dump(
            gs.best_estimator_, join(output_dir, "models", f"best_model_{tick}.joblib")
        )

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
@click.option("--parallel", is_flag=True)
@click.option("--n_workers", type=click.INT, default=16)
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
    parallel,
    n_workers,
):
    hparams = locals()
    random.seed(seed)

    create_dir(output_dir)
    create_dir(join(output_dir, "models"))

    in_dir = (
        join("data", "processed", "SP500_technical")
        if classifier != "L3"
        else join("data", "processed", "technical_discretized", "DENSE")
    )
    stocks = load_OHLCV_files(in_dir)
    logger.info(f"Loaded {len(stocks)} stocks")

    stocks = sorted(stocks, key=lambda s: s[0])

    if test_run:
        stocks = random.sample(stocks, k=3)

    ticks, _ = zip(*stocks)

    experiment = None
    if log_comet:
        exp = create_experiment()
        exp.add_tag("training")
        exp.log_parameters(hparams)
        exp.log_other("n_stocks", len(stocks))

    if parallel:
        results = Parallel(n_jobs=n_workers)(
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
                seed,
                experiment,
            )
            for tick, stock_df in tqdm(stocks, desc="Stocks")
        )
    else:
        results = [
            process_stock(
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
                seed,
                experiment,
            )
            for tick, stock_df in tqdm(stocks, desc="Stocks")
        ]

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

    return


if __name__ == "__main__":
    main()
