from src.config import MODELS, SCALE_FEATURES
from src.data_loader import load_data
from src.model import build_pipeline, evaluate_model, get_models, tune_model
from src.preprocessing import clean_data, feature_engineer, prepare_data
from src import config


def main():
    # 1. Load data
    print("Loading data...")
    df = load_data(config.DB_PATH, config.TABLE_NAME)
    print(f"  Loaded {len(df)} rows")

    # 2. Clean data
    print("Cleaning data...")
    df = clean_data(df)
    print(f"  {len(df)} rows after cleaning")

    # 3. Feature engineering
    print("Engineering features...")
    df = feature_engineer(df)

    # 4. Prepare data (encode + split)
    print("Preparing data (encoding and splitting)...")
    X_train, X_test, y_train, y_test = prepare_data(df)
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"  Features: {X_train.shape[1]}")

    # 5. Train and evaluate models
    models = get_models()
    all_results = {}

    for name, search_config in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        print(f"{'='*60}")

        model = models[name]
        pipeline = build_pipeline(model, SCALE_FEATURES, X_train.columns)

        print(f"  Search: {search_config['search']}", end="")
        if search_config["n_iter"]:
            print(f" (n_iter={search_config['n_iter']})", end="")
        print(f", CV={config.CV_FOLDS}, Scoring={config.SCORING}")

        search = tune_model(pipeline, search_config, X_train, y_train)
        print(f"  Best CV {config.SCORING}: {search.best_score_:.4f}")

        results = evaluate_model(search, X_test, y_test)
        all_results[name] = results

    # 6. Comparison table
    print(f"\n{'='*60}")
    print("MODEL COMPARISON (sorted by F1)")
    print(f"{'='*60}")
    header = f"{'Model':<25} {'Accuracy':>8} {'Precision':>9} {'Recall':>8} {'F1':>8} {'AUC-ROC':>8} {'CV F1':>8}"
    print(header)
    print("-" * len(header))

    sorted_results = sorted(all_results.items(), key=lambda x: x[1]["f1"], reverse=True)
    for name, r in sorted_results:
        print(
            f"{name:<25} {r['accuracy']:>8.4f} {r['precision']:>9.4f} "
            f"{r['recall']:>8.4f} {r['f1']:>8.4f} {r['roc_auc']:>8.4f} "
            f"{r['best_cv_score']:>8.4f}"
        )

    # 7. Best model
    best_name, best_r = sorted_results[0]
    print(f"\nBest Model: {best_name}")
    print(f"  F1: {best_r['f1']:.4f}, AUC-ROC: {best_r['roc_auc']:.4f}")
    print(f"  Best params: {best_r['best_params']}")


if __name__ == "__main__":
    main()
