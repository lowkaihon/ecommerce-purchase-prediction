DB_PATH = "data/online_shopping.db"
TABLE_NAME = "online_shopping"
TEST_SIZE = 0.2
RANDOM_STATE = 1
CV_FOLDS = 5
SCORING = "f1"

SCALE_FEATURES = [
    "ExitRate", "PageValue", "BounceRate", "ProductPageTime",
    "PageValue_log", "ProductPageTime_log",
]

CATEGORICAL_FEATURES = ["CustomerType", "TrafficSource", "GeographicRegion"]

MODELS = {
    "Logistic Regression": {
        "search": "grid",
        "n_iter": None,
        "params": {
            "model__C": [0.001, 0.01, 0.1, 1, 10, 100],
            "model__l1_ratio": [0, 0.25, 0.5, 0.75, 1],
            "model__solver": ["saga"],
        },
    },
    "LightGBM": {
        "search": "random",
        "n_iter": 50,
        "params": {
            "model__n_estimators": [100, 200, 500],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "model__max_depth": [3, 5, 7, -1],
            "model__num_leaves": [15, 31, 63],
            "model__min_child_samples": [5, 10, 20],
            "model__subsample": [0.7, 0.8, 1.0],
            "model__colsample_bytree": [0.7, 0.8, 1.0],
        },
    },
    "Random Forest": {
        "search": "random",
        "n_iter": 50,
        "params": {
            "model__n_estimators": [100, 200, 500],
            "model__max_depth": [5, 10, 15, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2"],
        },
    },
}
