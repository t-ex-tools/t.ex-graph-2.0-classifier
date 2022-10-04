import functions, config

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def read(path):
  df = pd.read_csv(path, usecols=functions.excluder)
  df.columns= df.columns.str.lower()
  return df

# NOTE: only for binary classification
#       where True/False are encoded as 1/0
def sample_equal_distribution(df, col):
  x = df[df[col] == 1]
  y = df[df[col] == 0]

  sample_size = x.size if x.size <= y.size else y.size
  return pd.concat([x.sample(n=sample_size, replace=True), y.sample(n=sample_size, replace=True)])

def split(X, y):
  X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, train_size=config.train_size)

  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)

  return X_train, X_test, y_train, y_test