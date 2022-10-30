from sklearn.preprocessing import StandardScaler

import export
import data
import report


def test(
    model, continuous, X, y, X_train, X_test, y_train, y_test, features, target, dataset, root
):

    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)

    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)

    model_name = type(model).__name__
    if dataset['smote'] is False:
      export.misclassifications(dataset, X_test, predictions, target, model_name, root)

    result = {
        "target": target,
        "train_test": None,
        "feature_importance": None,
        "cross_validation": None,
    }

    if continuous is True:
        result["train_test"] = report.continuous(y_test, predictions)
    else:
        result["train_test"] = report.category(y_test, predictions)

    result["feature_importance"] = report.feature_importance(
        model, X_test.values, y_test, features
    )

    result["cross_validation"] = report.cross_validation(model, X, y)
    
    output = dict()
    output[model_name] = result

    return output


def test_models_on_dataset(models, dataset, features, target, root):
    df = dataset['data']
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = data.split(X, y, dataset['smote'])

    continuous = y.dtype == "float64"
    output = dict()

    for model in models[target]:
        output = {
            **output,
            **test(
                model,
                continuous,
                X,
                y,
                X_train,
                X_test,
                y_train,
                y_test,
                features,
                target,
                dataset,
                root
            ),
        }

    return output


def compute_results(datasets, models, features, targets, root):
    results = {}

    for target in targets:
        for dataset in datasets:
            if dataset.get("label") not in results:
                results[dataset.get("label")] = dict()

            if target not in results[dataset.get("label")]:
                results[dataset.get("label")][target] = dict()

            results[dataset.get("label")][target] = {
                **results[dataset.get("label")][target],
                **test_models_on_dataset(models, dataset, features, target, root),
            }

    return results
