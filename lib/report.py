import config

import pandas as pd
from matplotlib import pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report
from sklearn.model_selection import KFold, cross_val_score
from sklearn.inspection import permutation_importance

def continuous(y_test, predictions):
  return {
    'r2': r2_score(y_test, predictions),
    'mse': mean_squared_error(y_test, predictions, squared=False),
    'mae': mean_absolute_error(y_test, predictions)
  }

def category(y_test, predictions):
  return classification_report(y_test, predictions, output_dict=True)

def feature_importance(model, X_test, y_test, features):
  result = permutation_importance(
    model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
  )
  importances = pd.Series(result.importances_mean, index=features)

  fig, ax = plt.subplots()
  importances.plot.bar(yerr=result.importances_std, ax=ax)
  fig.tight_layout()
  return plt

def cross_validation(model, X, y):
  kf = KFold(n_splits=config.k_fold, random_state=None) 
  result = cross_val_score(model , X, y, cv=kf)
  return format(result.mean())