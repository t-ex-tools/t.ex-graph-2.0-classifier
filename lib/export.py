from os import listdir, mkdir
from os.path import join, exists

import pandas as pd
import matplotlib.pyplot as plt

def classification_results(results, root):
  for key in results.keys():
    columns = {
      'continuous': [],
      'category': []    
    }

    index = {
      'continuous': [],
      'category': []
    }
    
    values = {
      'continuous': [],
      'category': []
    }

    for model in results.get(key).keys():
      is_continuous = results.get(key).get(model).get('continuous')
      pointer = 'continuous' if is_continuous is True else 'category'

      columns[pointer] = [key] + list(results.get(key).get(model).get('train_test').keys())
      index[pointer].append(model)
      values[pointer].append([model] + list(results.get(key).get(model).get('train_test').values()))
    
    for x in ['continuous', 'category']:
      df = pd.DataFrame(values[x], index=index[x], columns=columns[x])

      dir = join(root, 'results')
      if not exists(dir):
        mkdir(dir)

      filename = key.replace('/', '-') + '-' + x + '.csv'
      df.to_csv(join(dir, filename), sep=',')

def aggregated_classification_results(root):
  path = join(root, 'results')
  files = listdir(path)

  for y in ['continuous', 'category']:
    filtered = list(filter(lambda x: y in x, files))
    df = pd.concat([pd.read_csv(join(path, x)) for x in filtered], axis=1)
    df.drop(df.filter(regex="Unname"), axis=1, inplace=True)
    filename = 'aggregated-' + y + '.csv'
    df.to_csv(join(path, filename), sep=',')

def feature_importances(nrows, ncols, results, root):
    for key in results.keys():
      fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, figsize=(12, 10), dpi=300)

      for index, model in enumerate(results.get(key)):
        result = results.get(key).get(model).get('feature_importance').get('result')
        importances = results.get(key).get(model).get('feature_importance').get('importances')

        x = 0 if results.get(key).get(model).get('continuous') else 1  
        importances.plot.bar(yerr=result.importances_std, ax=ax[x, index % ncols])
        ax[x, index % ncols].set_title(model)
        
      ax[0, 0].set(ylabel='Predict weight')
      ax[1, 0].set(ylabel='Binary classification')
      fig.autofmt_xdate(rotation=45)
      
      filename = 'feature-importances-' + key + '.pdf'
      fig.savefig(join(root, 'results', filename))