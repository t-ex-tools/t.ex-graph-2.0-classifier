import numpy as np
import pandas as pd

import config


def read(path):
    df = pd.read_csv(
        path, usecols=lambda x: x.lower() not in config.excluded_cols, index_col=False
    )
    df.columns = df.columns.str.lower()

    return df


def feature_vector(df):
    return [
        col for col in list(df.columns) if col.lower() not in config.excluded_features
    ]


def binary_classification_labels(col, df, fn):
    df[col] = np.where(fn(df[config.tracking_ratio]), 1, 0)


def multi_classification_labels(col, df):
    df[col] = np.select(
        [x["condition"] for x in config.multi_labels(df)],
        [x["label"] for x in config.multi_labels(df)],
    )
