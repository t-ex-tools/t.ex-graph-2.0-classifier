import matplotlib.backends
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

import config


def continuous(y_test, predictions):
    return {
        "r2": r2_score(y_test, predictions),
        "mse": mean_squared_error(y_test, predictions, squared=False),
        "mae": mean_absolute_error(y_test, predictions),
    }


def category(y_test, predictions):
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, average="macro"),
        "recall": recall_score(y_test, predictions, average="macro"),
        "f1_score": f1_score(y_test, predictions, average="macro"),
    }


def feature_importance(model, X_test, y_test, features):
    result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
    )
    importances = pd.Series(result.importances_mean, index=features)

    return {"result": result, "importances": importances}


def cross_validation(model, X, y):
    kf = RepeatedStratifiedKFold(n_splits=config.k_fold, n_repeats=3, random_state=1)
    result = cross_val_score(model, X, y, cv=kf)
    return format(result.mean())
