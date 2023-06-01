import config
import export
import report

from imblearn.over_sampling import BorderlineSMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd

from os.path import join

def test(
    model,
    continuous,
    X,
    y,
    X_train,
    X_test,
    y_train,
    y_test,
    X_train_scaled,
    X_test_scaled,
    features,
    target,
    dataset,
    root,
    compute_options,
    no_train
):
    if no_train is False:
        model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)

    model_name = type(model).__name__
    export.model(model, dataset.get('label'), model_name, root)

    if compute_options["misclassifications"] is True:
        export.misclassifications(
            dataset, X_test, predictions, target, model_name, root
        )

    result = {
        "target": target,
        "train_test": None,
        "feature_importance": None,
        "cross_validation": None,
    }

    if compute_options["classifications"] is True:
        if continuous is True:
            result["train_test"] = report.continuous(y_test, predictions)
        else:
            result["train_test"] = report.category(y_test, predictions)

    if compute_options["feature_importances"] is True:
        result["feature_importance"] = report.feature_importance(
            model, X_test.values, y_test, features
        )

    if compute_options["cross_validations"] is True:
        result["cross_validation"] = report.cross_validation(model, X, y)

    output = dict()
    output[model_name] = result

    return output


def test_models_on_dataset(models, dataset, features, target, root, compute_options, no_train, scaler):
    df = dataset["data"]
    X = df[features]
    y = df[target]

    if no_train is False:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=config.train_size
        )

        if dataset["smote"] is True:
            X_train, y_train = BorderlineSMOTE().fit_resample(X_train, y_train)

        scaler = StandardScaler()    
        X_train_scaled = scaler.fit_transform(X_train)
        export.model(scaler, dataset.get('label'), 'std_scaler', root)
    else:
        X_train = None
        X_train_scaled = None
        y_train = None
        X_test = X
        y_test = y

    X_test_scaled = scaler.transform(X_test)

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
                X_train_scaled,
                X_test_scaled,
                features,
                target,
                dataset,
                root,
                compute_options,
                no_train
            ),
        }

    return output


def compute_results(datasets, models, features, targets, root, compute_options, no_train=False, scaler=None):
    results = {}

    for target in targets:
        for dataset in datasets:
            if dataset.get("label") not in results:
                results[dataset.get("label")] = dict()

            if target not in results[dataset.get("label")]:
                results[dataset.get("label")][target] = dict()

            results[dataset.get("label")][target] = {
                **results[dataset.get("label")][target],
                **test_models_on_dataset(
                    models, dataset, features, target, root, compute_options, no_train, scaler
                ),
            }

    if compute_options['classifications'] is True:
        export.classification_results(results, root)

    if compute_options['feature_importances'] is True:
        export.feature_importances(results, root)

    if compute_options['cross_validations'] is True:
        export.cross_validation(results, root)

    dir = join(
        root, 
        config.results_dir, 
        config.classifier_results_dir,
        config.binary_tracker
    )

    for dataset in datasets:
        df = pd.read_csv(
            join(dir, dataset['label'] + '.csv')
        )
        for col in ['accuracy', 'precision', 'recall', 'f1_score']:
            df[col] = df[col].apply(lambda x: '{:0.3f}'.format(x))
        
        df = df.sort_values(['accuracy'], ascending=[False])
        export.save_csv(df, dir, dataset['label'] + '-pretty')                  

    return results
