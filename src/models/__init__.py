import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from l3wrapper.l3wrapper import L3Classifier
import pandas as pd
import numpy as np
from fylearn.frr import FuzzyReductionRuleClassifier
from fylearn.fpt import FuzzyPatternTreeTopDownClassifier
from fylearn.garules import EnsembleMultimodalEvolutionaryClassifier


PARAMS_GRID = {
    "RFC": {
        "criterion": ["gini", "entropy"],
        "min_samples_split": [0.01, 0.05],
        "min_samples_leaf": [0.005, 0.01],
        "max_depth": [None, 5, 10, 20],
    },
    "MLP": [
        {
            "hidden_layer_sizes": [(100,)],
            "activation": ["relu", "tanh"],
            "solver": ["adam"],
            "learning_rate_init": [2e-5, 1e-4],
            "max_iter": [1000],
            "alpha": [1e-3, 1e-2],
            "batch_size": [1024],
            "early_stopping": [True],
            "n_iter_no_change": [10],
        },
        {
            "hidden_layer_sizes": [(100,)],
            "activation": ["relu", "tanh"],
            "learning_rate_init": [2e-5, 1e-4],
            "solver": ["lbfgs"],
            "max_iter": [1000],
            "alpha": [1e-3, 1e-2],
            "batch_size": [1024],
        },
    ],
    "SVC": [
        {
            "kernel": ["poly"],
            "degree": [3, 4, 5],
            "C": [1e-3, 1e-2, 1e-1, 10, 100],
        },
        {
            "kernel": ["rbf"],
            "C": [1e-3, 1e-2, 1e-1, 10, 100],
        },
    ],
    "LG": [
        {
            "solver": ["liblinear", "saga"],
            "penalty": ["l1", "l2"],
            "C": [1e-4, 1e-3, 1e-2, 1e-1, 1, 10],
        },
        {
            "solver": ["newton-cg", "sag", "lbfgs"],
            "penalty": ["l2"],
            "C": [1e-4, 1e-3, 1e-2, 1e-1, 1, 10],
        },
    ],
    "KNN": {
        "weights": ["uniform", "distance"],
        "n_neighbors": [3, 5, 7],
        "algorithm": ["ball_tree", "kd_tree"],
    },
    "GNB": {},
    "L3": {
        "min_sup": [0.005, 0.01, 0.05, 0.1],
        "min_conf": [0.5, 0.75],
        "max_matching": [1, 3],
        # "max_length": [0, 5],
    },
    "FRR": {}, # FuzzyReductionRuleClassifier
    "FPTTD": {}, # FuzzyPatternTreeTopDownClassifier
    "EMEC": {} # EnsembleMultimodalEvolutionaryClassifier
}


PARAMS = {
    "RFC": {"criterion": "gini", "min_samples_split": 0.01, "min_samples_leaf": 0.005},
    "KNN": {"weights": "distance", "n_neighbors": 3, "algorithm": "ball_tree"},
    "MLP": {
        "hidden_layer_sizes": (10, 10),
        "activation": "logistic",
        "solver": "lbfgs",
        "learning_rate": "constant",
        "learning_rate_init": 0.0001,
    },
    "SVC": {"kernel": "poly", "degree": 4, "C": 50},
    "LG": {
        "solver": "liblinear",
        "penalty": "l1",
        "class_weight": "balanced",
        "C": 1,
    },
    "GNB": {},
    "L3": {
        "min_sup": 0.005,  # [0.005, 0.01, 0.05, 0.1],
        "min_conf": 0.5,
        "max_matching": 1,
        "max_length": 0,
    },
    "FRR": {}, # FuzzyReductionRuleClassifier
    "FPTTD": {}, # FuzzyPatternTreeTopDownClassifier
    "EMEC": {} # EnsembleMultimodalEvolutionaryClassifier
}

FUZZY_CLASSIFIERS = ["FRR", "FPTTD", "EMEC"]

def instantiate_classifier(
    classifier, return_grid=False, load_default=False, **kw_classifier
):
    """Instantiate and return a ML model, along with its parameters.

    Optionally, return the grid of parameters for grid search
    """

    if load_default:
        params = PARAMS[classifier]
    else:
        params = dict()

    params.update(kw_classifier)
    N_JOBS = -1

    if classifier == "KNN":
        clf = KNeighborsClassifier(n_jobs=N_JOBS, **params)
    elif classifier == "RFC":
        clf = RandomForestClassifier(
            n_jobs=N_JOBS,
            n_estimators=200,
            class_weight="balanced_subsample",
            **params,
        )
    elif classifier == "SVC":
        clf = SVC(gamma="scale", class_weight="balanced", **params)
    elif classifier == "MLP":
        clf = MLPClassifier(
            **params,
        )
    elif classifier == "L3":
        clf = L3Classifier(**params)
    elif classifier == "LG":
        clf = LogisticRegression(n_jobs=N_JOBS, **params)
    elif classifier == "GNB":
        clf = GaussianNB()
    elif classifier == "FRR": # FuzzyReductionRuleClassifier
        clf = FuzzyReductionRuleClassifier()
    elif classifier == "FPTTD": # FuzzyPatternTreeTopDownClassifier
        clf = FuzzyPatternTreeTopDownClassifier()
    elif classifier == "EMEC": # EnsembleMultimodalEvolutionaryClassifier
        clf = EnsembleMultimodalEvolutionaryClassifier()
    else:
        raise NotImplementedError()

    return (clf, params, PARAMS_GRID[classifier]) if return_grid else (clf, params)



def get_scores_filename(tick):
    return f"scores_{tick}.csv"


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
