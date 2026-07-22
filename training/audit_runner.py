import os
import sys
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

# Resolve paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from training.splits import split_dataset
from training.ablation import run_ablation_study
from training.calibration import evaluate_calibration
from training.representation_audit import run_representation_audit

def main():
    print("=" * 60)
    print("   RUNNING SPRINT 5 SCIENTIFIC AUDIT & REPRESENTATION STUDY")
    print("=" * 60)

    # 1. Load Parquet dataset
    parquet_path = PROJECT_ROOT / "matches_with_features.parquet"
    if not parquet_path.exists():
        print(f"Error: Parquet file not found at {parquet_path}. Run preprocessing first.")
        sys.exit(1)

    print(f"Loading dataset from: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    # 2. Chronological split
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)

    # 3. Part 2: Feature Ablation Study
    print("\n[Step 1] Running Feature Ablation Study...")
    ablation_results = run_ablation_study(X_train, X_val, X_test, y_train, y_val, y_test)
    for group, metrics in ablation_results.items():
        print(f"  • {group}:")
        print(f"    - LR Acc:  {metrics['lr_accuracy']:.4f} (Degradation: {metrics['lr_degradation']:.4f})")
        print(f"    - XGB Acc: {metrics['xgb_accuracy']:.4f} (Degradation: {metrics['xgb_degradation']:.4f})")

    # 4. Part 3: Probability Calibration
    print("\n[Step 2] Evaluating Probability Calibration...")
    # We retrain our best benchmark model from Sprint 4: XGBoost to analyze calibration
    xgb = XGBClassifier(
        n_estimators=100, 
        max_depth=4, 
        learning_rate=0.05, 
        eval_metric="logloss", 
        random_state=42, 
        n_jobs=-1
    )
    xgb.fit(X_train.astype(float), y_train)
    probs = xgb.predict_proba(X_test.astype(float))[:, 1]
    
    calibration_results = evaluate_calibration(y_test.values, probs, n_bins=10)
    print(f"  • Brier Score:                    {calibration_results['brier_score']:.4f}")
    print(f"  • Expected Calibration Error (ECE): {calibration_results['ece']:.4f}")
    print(f"  • Reliability Bins:")
    for b in calibration_results["bin_statistics"][:5]:
        print(f"    - Bin {b['bin']}: Conf: {b['confidence']:.4f} | Obs Win: {b['accuracy']:.4f} ({b['samples']} samples)")

    # 5. Part 4: Statistical Confidence / Variance
    print("\n[Step 3] Quantifying Statistical Confidence (Cross-Validation)...")
    # Standardize all columns to float for cross_val_score
    X_all = pd.concat([X_train, X_val, X_test]).astype(float)
    y_all = pd.concat([y_train, y_val, y_test])
    
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    t0 = time.time()
    cv_scores = cross_val_score(xgb, X_all, y_all, cv=cv, scoring="accuracy", n_jobs=-1)
    cv_duration = time.time() - t0
    
    mean_score = float(np.mean(cv_scores))
    std_score = float(np.std(cv_scores))
    ci_lower = mean_score - (1.96 * std_score)
    ci_upper = mean_score + (1.96 * std_score)
    
    print(f"  • 10-fold CV Accuracy: {mean_score*100:.2f}% +/- {std_score*100:.2f}%")
    print(f"  • 95% Confidence Interval: [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]")
    print(f"  • CV Time: {cv_duration:.2f} seconds")

    # 6. Part 5: Representation Audit & Hypothesis Testing
    print("\n[Step 4] Auditing Feature Representation & Compression...")
    representation_results = run_representation_audit(df, xgb, X_test)
    rep = representation_results["deck_representation"]
    print(f"  • Unique Decks Checked:           {rep['unique_decks']:,}")
    print(f"  • Unique Feature Vectors:         {rep['unique_feature_vectors']:,}")
    print(f"  • Compression Ratio:              {rep['compression_ratio']:.4f}")
    print(f"  • Information / Identity Loss:    {rep['identity_loss_percentage']:.2f}%")
    
    counter = representation_results["counter_matchups"]
    print(f"  • Counter Matchup Analysis (Giant vs Inferno Tower):")
    print(f"    - Samples:                      {counter['samples']}")
    print(f"    - Model Accuracy on Slices:     {counter['model_accuracy']*100:.2f}%")
    print(f"    - Observed Win Rate on Slices:  {counter['observed_win_rate']*100:.2f}%")

    # 7. Save Consolidated Audit Results
    audit_report = {
        "ablation_results": ablation_results,
        "calibration_results": calibration_results,
        "statistical_confidence": {
            "cv_scores": cv_scores.tolist(),
            "mean_accuracy": mean_score,
            "std_dev": std_score,
            "confidence_interval": [ci_lower, ci_upper]
        },
        "representation_audit": representation_results
    }

    report_path = PROJECT_ROOT / "logs" / "audit_results.json"
    with open(report_path, "w") as f:
        json.dump(audit_report, f, indent=4)
        
    print(f"\n[+] Audit runner completed! Results logged in {report_path}\n")

if __name__ == "__main__":
    main()
