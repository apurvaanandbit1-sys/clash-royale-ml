import os
import sys
import json
import pandas as pd
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from training.dataset_validation import validate_dataset
from training.splits import split_dataset
from training.eda import perform_eda
from training.benchmarks import train_and_evaluate_benchmarks
from training.error_analysis import analyze_prediction_errors

def run_ml_pipeline():
    print("=" * 60)
    print("   RUNNING BENCHMARK MACHINE LEARNING PIPELINE")
    print("=" * 60)

    # 1. Load Parquet dataset
    parquet_path = PROJECT_ROOT / "matches_with_features.parquet"
    if not parquet_path.exists():
        print(f"Error: Parquet file not found at {parquet_path}. Run preprocessing first.")
        sys.exit(1)

    print(f"Loading dataset from: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    # 2. Dataset Validation
    print("\n[Step 1] Auditing Dataset Integrity...")
    validation_results = validate_dataset(df)
    print(f"  • Row count: {validation_results['row_count']:,}")
    print(f"  • Feature count: {validation_results['feature_count']}")
    print(f"  • Duplicate rows: {validation_results['duplicate_rows_count']}")
    print(f"  • Duplicate matchups: {validation_results['duplicate_matchups_count']}")
    print(f"  • Class distribution: {validation_results['class_balance']}")
    print(f"  • Invalid values violations: {validation_results['invalid_values']}")

    # 3. Exploratory Data Analysis
    print("\n[Step 2] Performing Statistical EDA...")
    eda_results = perform_eda(df)
    print(f"  • Class Balance Win Ratio: {eda_results['class_balance']['win_ratio']*100:.2f}%")
    print(f"  • Redundant feature pairs (|r| > 0.8): {len(eda_results['redundant_pairs'])}")
    for pair in eda_results['redundant_pairs'][:3]:
        print(f"    - {pair['feature_1']} <-> {pair['feature_2']} (r = {pair['correlation']:.2f})")
    if len(eda_results['redundant_pairs']) > 3:
        print(f"    - ... and {len(eda_results['redundant_pairs'])-3} more pairs.")

    # 4. Partition Splits
    print("\n[Step 3] Chronological Splitting...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)

    # 5. Train & Evaluate Benchmark Models
    print("\n[Step 4] Training Benchmark Models...")
    dataset_version = f"clashroyale_db_v1_{len(df)}_rows"
    benchmark_results = train_and_evaluate_benchmarks(
        X_train, X_val, X_test, y_train, y_val, y_test, 
        dataset_version=dataset_version,
        random_seed=42
    )

    # 6. Extract Best Model for Error & Feature Importance Audits
    # Find model with highest test accuracy
    best_model_name = "Dummy (Baseline)"
    best_acc = 0.0
    for name, res in benchmark_results.items():
        acc = res["metrics"]["test_accuracy"]
        if acc > best_acc:
            best_acc = acc
            best_model_name = name

    print(f"\n[Step 5] Selecting Best Performer for Error Auditing: {best_model_name} (Acc: {best_acc*100:.2f}%)")
    best_model = benchmark_results[best_model_name]["model"]
    
    # Scale inputs for scaling-sensitive models
    if best_model_name in ["Logistic Regression", "MLP (Neural Net)"]:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(X_train)
        test_x = scaler.transform(X_test)
    else:
        test_x = X_test.astype(float)
        
    predictions = best_model.predict(test_x)
    if hasattr(best_model, "predict_proba"):
        probabilities = best_model.predict_proba(test_x)[:, 1]
    else:
        probabilities = predictions
        
    error_results = analyze_prediction_errors(X_test, y_test, predictions, probabilities)
    
    print("\n[Error Analysis] Prediction Error Rates:")
    print("  • Trophy Slices:")
    for bucket, stats in error_results.get("trophy_errors", {}).items():
        print(f"    - {bucket}: {stats['error_rate']*100:.2f}% error rate ({stats['samples']} samples)")
    print("  • Top Arenas:")
    for arena, stats in error_results.get("arena_errors", {}).items():
        print(f"    - {arena}: {stats['error_rate']*100:.2f}% error rate ({stats['samples']} samples)")
    print("  • Mirror Slices (has_mirror):")
    for has_m, stats in error_results.get("mirror_errors", {}).items():
        print(f"    - {has_m}: {stats['error_rate']*100:.2f}% error rate ({stats['samples']} samples)")

    # 7. Feature Importance Audit (XGBoost and Random Forest)
    print("\n[Step 6] Auditing Feature Importances...")
    feature_audit = {}
    
    if "Random Forest" in benchmark_results:
        rf_model = benchmark_results["Random Forest"]["model"]
        rf_importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
        print("  • Top 5 Random Forest Gini Importances:")
        top_rf = rf_importances.sort_values(ascending=False).head(5)
        for f, val in top_rf.items():
            print(f"    - {f}: {val:.4f}")
        feature_audit["random_forest"] = top_rf.to_dict()
        
    if "XGBoost" in benchmark_results:
        xgb_model = benchmark_results["XGBoost"]["model"]
        xgb_importances = pd.Series(xgb_model.feature_importances_, index=X_train.columns)
        print("  • Top 5 XGBoost Feature Importances:")
        top_xgb = xgb_importances.sort_values(ascending=False).head(5)
        for f, val in top_xgb.items():
            print(f"    - {f}: {val:.4f}")
        feature_audit["xgboost"] = top_xgb.to_dict()

    # 8. Save final aggregated results
    pipeline_report = {
        "validation_results": validation_results,
        "eda_results": {
            "redundant_pairs": eda_results["redundant_pairs"],
            "class_balance": eda_results["class_balance"]
        },
        "benchmark_scores": {
            name: {
                "metrics": res["metrics"],
                "train_time": res["train_time"],
                "inference_latency": res["inference_latency"]
            } for name, res in benchmark_results.items()
        },
        "error_analysis": error_results,
        "feature_audit": feature_audit
    }

    report_path = PROJECT_ROOT / "logs" / "pipeline_results.json"
    with open(report_path, "w") as f:
        json.dump(pipeline_report, f, indent=4)
        
    print(f"\n[+] Pipeline execution completed! Results stored in {report_path}\n")

if __name__ == "__main__":
    run_ml_pipeline()
