import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from l3wrapper.l3wrapper import L3Classifier
import pandas as pd
import numpy as np


PARAMS_GRID = {
    "RFC": {
        "criterion": ["gini", "entropy"],
        "min_samples_split": [0.01, 0.05],
        "min_samples_leaf": [0.005, 0.01],
    },
    "KNN": {
        "weights": ["uniform", "distance"],
        "n_neighbors": [3, 5, 7],
        "algorithm": ["ball_tree", "kd_tree"],
    },
    "MLP": {
        "hidden_layer_sizes": [(10,), (30,), (10, 10)],
        "activation": ["relu", "logistic", "tanh"],
        "solver": ["lbfgs", "sgd", "adam"],
        "learning_rate": ["constant", "invscaling"],
        "learning_rate_init": [0.0001, 0.001, 0.01, 0.1],
    },
    "SVC": {
        "kernel": ["linear", "poly", "rbf"],
        "degree": [3, 4, 5],
        "C": [0.001, 0.01, 1, 10, 50],
    },
    "L3": {
        "min_sup": [0.005],  # [0.005, 0.01, 0.05, 0.1],
        "min_conf": [0.5, 0.25, 0.75],
        "max_matching": [1, 3],
        "max_length": [0, 5],
    },
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
    "L3": {
        "min_sup": 0.005,  # [0.005, 0.01, 0.05, 0.1],
        "min_conf": 0.5,
        "max_matching": 1,
        "max_length": 0,
    },
}


def instantiate_classifier(classifier, return_grid=False, **kw_classifier):
    """Instantiate and return a ML model, along with its parameters.

    Optionally, return the grid of parameters for grid search
    """
    params = PARAMS[classifier]
    params.update(kw_classifier)

    if classifier == "KNN":
        clf = KNeighborsClassifier(n_jobs=-1, **params)
    elif classifier == "RFC":
        clf = RandomForestClassifier(
            n_jobs=-1,
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            **params,
        )
    elif classifier == "SVC":
        clf = SVC(gamma="scale", random_state=42, class_weight="balanced", **params)
    elif classifier == "MLP":
        clf = MLPClassifier(
            random_state=42,
            max_iter=10000,
            early_stopping=True,
            n_iter_no_change=3,
            **params,
        )
    elif classifier == "L3":
        clf = L3Classifier(**params)
    else:
        raise NotImplementedError()

    return (clf, params, PARAMS_GRID[classifier]) if return_grid else (clf, params)


def get_scores_filename(tick):
    return f"scores_{tick}.csv"


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)