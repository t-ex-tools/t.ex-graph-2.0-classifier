from os import listdir, mkdir, remove
from os.path import join, exists

import config

import pandas as pd
import matplotlib.pyplot as plt


def mkdir_p(path):
    if not exists(path):
        mkdir(path)


def reset_dir(dir):
    if not exists(dir):
        mkdir(dir)
    else:
        for file in listdir(dir):
            remove(join(dir, file))


def save_csv(df, path, dataset_name):
    df.drop(df.filter(regex="Unname"), axis=1, inplace=True)
    mkdir_p(path)

    filename = dataset_name.replace("/", "-") + ".csv"
    df.to_csv(join(path, filename), sep=",")


def classification_results(results, root):
    dir = join(root, config.results_dir, config.classifier_results_dir)
    reset_dir(dir)

    for key in results.keys():
        for target in results.get(key):
            columns = list()
            index = list()
            values = list()

            models = results.get(key).get(target).keys()
            for model in models:
                scores = results.get(key).get(target).get(model).get("train_test")
                columns = [key] + list(scores.keys())
                index.append(model)
                values.append([model] + list(scores.values()))

            if len(models) > 0:
                df = pd.DataFrame(values, index=index, columns=columns)
                save_csv(df, join(dir, target), key)


def feature_importances(results, root):
    dir = join(root, config.results_dir, config.feature_importances_dir)
    reset_dir(dir)

    for key in results.keys():
        for target in results.get(key):
            ncols = len(results.get(key).get(target).keys())
            if ncols < 1:
                return

            fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(48, 8), dpi=300)

            for index, model in enumerate(results.get(key).get(target)):
                scores = (
                    results.get(key).get(target).get(model).get("feature_importance")
                )
                result = scores.get("result")
                importances = scores.get("importances")

                ax = axes[index % ncols] if ncols > 1 else axes
                importances.plot.bar(yerr=result.importances_std, ax=ax)
                ax.set_title(model)

            fig.autofmt_xdate(rotation=45)

            path = join(dir, target)
            mkdir_p(path)

            filename = key.replace("/", "-") + ".pdf"
            fig.savefig(join(path, filename))


def cross_validation(results, root):
    dir = join(root, config.results_dir, config.cross_validation_dir)
    reset_dir(dir)

    for key in results.keys():
        for target in results.get(key):
            columns = list()
            index = list()
            values = list()

            models = results.get(key).get(target).keys()
            for model in models:
                score = results.get(key).get(target).get(model).get("cross_validation")
                columns = [key, "RepeatedStratifiedKFold with K=" + str(config.k_fold)]
                index.append(model)
                values.append([model, score])

            if len(models) > 0:
                df = pd.DataFrame(values, index=index, columns=columns)
                save_csv(df, join(dir, target), key)
