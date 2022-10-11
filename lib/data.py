import config

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def read(path):
    df = pd.read_csv(path, usecols=lambda x: x.lower() not in config.excluded_cols)
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


def split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=True, train_size=config.train_size
    )

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_train, X_test, y_train, y_test


def feature_vector(df):
    return [
        col for col in list(df.columns) if col.lower() not in config.excluded_features
    ]


def binary_classification_labels(col, df):
    df[col] = np.where(df["weight"] > 0, 1, 0)


def multi_classification_labels(col, df):
    df[col] = np.select(
        [x["condition"] for x in config.multi_labels(df)],
        [x["label"] for x in config.multi_labels(df)],
    )
