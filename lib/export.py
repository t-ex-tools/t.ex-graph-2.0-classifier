from os import listdir, makedirs, remove
from os.path import exists, join
from joblib import dump

import config
import matplotlib.pyplot as plt
import pandas as pd


def mkdir_p(path):
    if not exists(path):
        makedirs(path, exist_ok=True)


def reset_dir(dir):
    if not exists(dir):
        makedirs(dir, exist_ok=True)
    else:
        for file in listdir(dir):
            remove(join(dir, file))


def save_csv(df, path, dataset_name):
    df.drop(df.filter(regex="Unname"), axis=1, inplace=True)
    mkdir_p(path)

    filename = dataset_name.replace("/", "-") + ".csv"
    df.to_csv(join(path, filename), sep=",", index=False)


def misclassifications(dataset, X_test, predictions, target, model, root):
    pred = pd.DataFrame(
        data=predictions, columns=[config.pred_col], index=X_test.index.copy()
    )
    df = dataset["data"].merge(pred, how="left", left_index=True, right_index=True)
    df = df[df[config.pred_col].notnull()]
    df = df[df[target] != df[config.pred_col]]
    df = df[["id"] + config.included_features + [target, config.pred_col]]
    path = join(root, config.results_dir, config.misclassifications_dir, target, model)
    mkdir_p(path)
    filename = "misclassifications-" + dataset["label"].replace("/", "-") + ".csv"
    df.to_csv(join(path, filename))


def classification_results(results, root):
    dir = join(root, config.results_dir, config.classifier_results_dir)

    for key in results.keys():
        for target in results.get(key):
            columns = list()
            values = list()

            models = results.get(key).get(target).keys()
            for model in models:
                scores = results.get(key).get(target).get(model).get("train_test")
                columns = [key] + list(scores.keys())
                values.append([model] + list(scores.values()))

            if len(models) > 0:
                df = pd.DataFrame(values, columns=columns)
                save_csv(df, join(dir, target), key)


def feature_importances(results, root):
    dir = join(root, config.results_dir, config.feature_importances_dir)

    for key in results.keys():
        for target in results.get(key):
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(48, 8), dpi=300)

            for index, model in enumerate(results.get(key).get(target)):
                scores = (
                    results.get(key).get(target).get(model).get("feature_importance")
                )
                result = scores.get("result")
                importances = scores.get("importances")

                importances.plot.bar(yerr=result.importances_std, ax=axes)
                axes.set_title("Feature importances for " + model + " on " + key)
                axes.set_ylabel("Mean decrease in impurity")

                fig.autofmt_xdate(rotation=45)

                path = join(dir, target, model)
                mkdir_p(path)

                filename = key.replace("/", "-") + ".pdf"
                fig.savefig(join(path, filename))


def cross_validation(results, root):
    dir = join(root, config.results_dir, config.cross_validation_dir)

    for key in results.keys():
        for target in results.get(key):
            columns = list()
            values = list()

            models = results.get(key).get(target).keys()
            for model in models:
                score = results.get(key).get(target).get(model).get("cross_validation")
                columns = [key, "RepeatedStratifiedKFold with K=" + str(config.k_fold)]
                values.append([model, score])

            if len(models) > 0:
                df = pd.DataFrame(values, columns=columns)
                save_csv(df, join(dir, target), key)

def model(model, dataset_label, model_name, root):
    path = join(root, 'models', dataset_label)
    mkdir_p(path)
    dump(model, join(path, model_name + '.sav'))
