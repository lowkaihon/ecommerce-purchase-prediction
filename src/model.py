import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import CV_FOLDS, RANDOM_STATE, SCORING


def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=5000,
            random_state=RANDOM_STATE,
        ),
        "LightGBM": lgb.LGBMClassifier(
            objective="binary",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            verbose=-1,
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }


def build_pipeline(model, scale_features, X_columns):
    # Only scale columns that actually exist in the data
    valid_scale = [c for c in scale_features if c in X_columns]
    preprocessor = ColumnTransformer(
        [("scale", StandardScaler(), valid_scale)],
        remainder="passthrough",
    ).set_output(transform="pandas")
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def tune_model(pipeline, search_config, X_train, y_train):
    params = search_config["params"]
    if search_config["search"] == "grid":
        search = GridSearchCV(
            pipeline, params, cv=CV_FOLDS, scoring=SCORING, n_jobs=-1
        )
    else:
        search = RandomizedSearchCV(
            pipeline,
            params,
            n_iter=search_config["n_iter"],
            cv=CV_FOLDS,
            scoring=SCORING,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    search.fit(X_train, y_train)
    return search


def evaluate_model(search, X_test, y_test):
    y_pred = search.best_estimator_.predict(X_test)
    y_prob = search.best_estimator_.predict_proba(X_test)[:, 1]

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "best_params": search.best_params_,
        "best_cv_score": search.best_score_,
    }

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return results
