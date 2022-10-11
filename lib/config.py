binary_tracker = "binary_tracker"
multi_tracker = "multi_tracker"

excluded_cols = [
    "label",
    "timeset",
    "weighted indegree",
    "weighted outdegree",
    "weighted degree",
]
excluded_features = ["id", "weight", binary_tracker, multi_tracker]

train_size = 0.2
k_fold = 10


def multi_labels(df):
    return [
        {"label": 0, "condition": (df["weight"] <= 0.2)},
        {"label": 1, "condition": (df["weight"] > 0.2) & (df["weight"] <= 0.4)},
        {"label": 2, "condition": (df["weight"] > 0.4) & (df["weight"] <= 0.6)},
        {"label": 3, "condition": (df["weight"] > 0.6) & (df["weight"] <= 0.8)},
        {"label": 4, "condition": (df["weight"] > 0.8)},
    ]


results_dir = "results"
classifier_results_dir = "classification"
feature_importances_dir = "feature_importances"
