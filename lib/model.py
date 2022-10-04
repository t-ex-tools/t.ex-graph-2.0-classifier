import functions, report, data

def test(model, continuous, X, y, X_train, X_test, y_train, y_test, features):
  model.fit(X_train, y_train)
  predictions = model.predict(X_test)

  result = {
    "train_test": None,
    "feature_importance": None,
    "cross_validation": None
  }

  if continuous is True:
    result["train_test"] = report.continuous(y_test, predictions)
  else:
    result["train_test"] = report.category(y_test, predictions)

  result["feature_importance"] = report.feature_importance(model, X_test, y_test, features)
  result["cross_validation"] = report.cross_validation(model, X, y)

  output = dict()
  output[type(model).__name__] = result
  output['continuous'] = continuous
  return output

def test_models_on_dataset(models, dataset, features, target):
  X = dataset[features]
  y = dataset[target]

  continuous = functions.is_continuous(y.dtype)
  m = models['continuous'] if continuous else models['category']
  X_train, X_test, y_train, y_test = data.split(X, y)

  output = dict()
  
  for model in m:
    output = { **output, **test(model, continuous, X, y, X_train, X_test, y_train, y_test, features) }
  
  return output

def compute_results(datasets, models, features, targets):
  results = {}

  for target in targets:
    for dataset in datasets:
      if dataset.get('label') not in results:
        results[dataset.get('label')] = dict()
        
      results[dataset.get('label')] = { **results[dataset.get('label')], **test_models_on_dataset(models, dataset.get('data'), features, target) }

  return results