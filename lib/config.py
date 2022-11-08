tracking_ratio = "tracking"
binary_tracker = "binary_tracker"
multi_tracker = "multi_tracker"
pred_col = 'prediction'

excluded_cols = [
    "label",
    "timeset",
    "weighted indegree",
    "weighted outdegree",
    "weighted degree"
]
excluded_features = [
  "id", 
  "tracking", 
  tracking_ratio, 
  binary_tracker, 
  multi_tracker
]

included_features = [
  "count",
  tracking_ratio
]

train_size = 0.8
k_fold = 10


def multi_labels(df):
    return [
        {"label": 0, "condition": (df[tracking_ratio] <= 0.2)},
        {
            "label": 1,
            "condition": (df[tracking_ratio] > 0.2) & (df[tracking_ratio] <= 0.4),
        },
        {
            "label": 2,
            "condition": (df[tracking_ratio] > 0.4) & (df[tracking_ratio] <= 0.6),
        },
        {
            "label": 3,
            "condition": (df[tracking_ratio] > 0.6) & (df[tracking_ratio] <= 0.8),
        },
        {"label": 4, "condition": (df[tracking_ratio] > 0.8)},
    ]


results_dir = "results"
classifier_results_dir = "classification"
feature_importances_dir = "feature_importances"
cross_validation_dir = "cross_validation"
misclassifications_dir = "misclassifications"
