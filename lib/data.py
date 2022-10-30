import numpy as np
import pandas as pd
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.model_selection import train_test_split

import config


def read(path):
    df = pd.read_csv(
        path, usecols=lambda x: x.lower() not in config.excluded_cols, index_col=False
    )
    df.columns = df.columns.str.lower()

    return df


# NOTE: only for binary classification
#       where True/False are encoded as 1/0
def sample_equal_distribution(df, col):
    x = df[df[col] == 1]
    y = df[df[col] == 0]

    sample_size = x.size if x.size <= y.size else y.size
    return pd.concat(
        [x.sample(n=sample_size, replace=True), y.sample(n=sample_size, replace=True)]
    )


def split(X, y, smote):
    if smote is True:
      X, y = BorderlineSMOTE().fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=config.train_size
    )

    return X_train, X_test, y_train, y_test


def feature_vector(df):
    return [
        col for col in list(df.columns) if col.lower() not in config.excluded_features
    ]


def binary_classification_labels(col, df):
    df[col] = np.where(df[config.tracking_ratio] > 0, 1, 0)


def multi_classification_labels(col, df):
    df[col] = np.select(
        [x["condition"] for x in config.multi_labels(df)],
        [x["label"] for x in config.multi_labels(df)],
    )
