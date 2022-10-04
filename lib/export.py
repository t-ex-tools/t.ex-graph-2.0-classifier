import os
import os
from os.path import join
import pandas as pd

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

      columns[pointer] = results.get(key).get(model).get('train_test').keys()
      index[pointer].append(model)
      values[pointer].append(results.get(key).get(model).get('train_test').values())
    
    for x in ['continuous', 'category']:
      df = pd.DataFrame(values[x], index=index[x], columns=columns[x])

      dir = join(root, 'results')
      if not os.path.exists(dir):
        os.mkdir(dir)

      filename = key.replace('/', '-') + '-' + x + '.csv'
      df.to_csv(join(dir, filename), sep=',')