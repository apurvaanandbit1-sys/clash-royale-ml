import os
import sys
import json
import time
import yaml
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Resolve paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from training.dataset import get_dataloaders
from training.baselines import (
    RandomGuessModel, MajorityClassModel, LogisticRegressionModel,
    RandomForestModel, LightGBMModel, CatBoostModel, MLPModel
)

def build_one_hot_vectors(loader, num_cards: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Transforms PyTorch dataset indices and trophy differences 
    into a 245-dimensional continuous NumPy array.
    """
    X_list = []
    y_list = []
    
    for batch in loader:
        p1_deck = batch["p1_deck"]
        p2_deck = batch["p2_deck"]
        t_diff = batch["trophy_diff"]
        target = batch["target"]
        
        # Convert deck card indices [0, C-1] to multi-hot [Batch, C]
        p1_one_hot = torch.nn.functional.one_hot(p1_deck, num_classes=num_cards).sum(dim=1)
        p2_one_hot = torch.nn.functional.one_hot(p2_deck, num_classes=num_cards).sum(dim=1)
        
        # Concatenate player 1, player 2, and trophy difference
        feat = torch.cat([p1_one_hot, p2_one_hot, t_diff], dim=1)
        X_list.append(feat.numpy())
        y_list.append(target.numpy())
        
    return np.vstack(X_list), np.vstack(y_list).squeeze()

def run_baselines_benchmark():
    print("=" * 60)
    print("   RUNNING SPRINT 11 BASELINE MODELS BENCHMARK")
    print("=" * 60)

    # 1. Load configurations
    with open(PROJECT_ROOT / "config" / "dataset_config.yaml", "r") as f:
        ds_config = yaml.safe_load(f)
    with open(PROJECT_ROOT / "config" / "benchmark_config.yaml", "r") as f:
        m_config = yaml.safe_load(f)

    # 2. Get DataLoaders from Sprint 10
    print("\n[Step 1] Loading Parquet Dataset & Building loaders...")
    df = pd.read_parquet(PROJECT_ROOT / ds_config["data"]["dataset_path"])
    
    # Load unique card ids to get num_cards
    with open(PROJECT_ROOT / ds_config["data"]["card_library_path"], "r") as f:
        cards_lib = json.load(f)
    num_cards = len(cards_lib)

    train_ldr, val_ldr, test_ldr = get_dataloaders(
        df=df,
        card_library_path=PROJECT_ROOT / ds_config["data"]["card_library_path"],
        batch_size=ds_config["loader"]["batch_size"],
        seed=ds_config["loader"]["seed"],
        augment_prob=ds_config["augmentation"]["probability"]
    )

    # 3. Build One-hot features
    print("\n[Step 2] Translating batched indices into one-hot features...")
    # Note: Symmetry augmentation is active only on train_ldr
    X_train, y_train = build_one_hot_vectors(train_ldr, num_cards)
    X_val, y_val = build_one_hot_vectors(val_ldr, num_cards)
    X_test, y_test = build_one_hot_vectors(test_ldr, num_cards)

    print(f"  • Train split dimensions:      {X_train.shape}")
    print(f"  • Validation split dimensions: {X_val.shape}")
    print(f"  • Test split dimensions:       {X_test.shape}")

    # Standard scale only the trophy difference (column index 244)
    # The first 244 columns are binary 0/1, scaling them harms tree-based models and MLP.
    # We will standardize just the final column for MLP and LR
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[:, -1] = scaler.fit_transform(X_train[:, -1].reshape(-1, 1)).squeeze()
    X_val_scaled[:, -1] = scaler.transform(X_val[:, -1].reshape(-1, 1)).squeeze()
    X_test_scaled[:, -1] = scaler.transform(X_test[:, -1].reshape(-1, 1)).squeeze()

    # 4. Instantiate baselines
    baselines = {
        "Random Guess": (RandomGuessModel(), False),
        "Majority Predictor": (MajorityClassModel(), False),
        "Logistic Regression": (LogisticRegressionModel(
            C=m_config["models"]["logistic_regression"]["C"],
            max_iter=m_config["models"]["logistic_regression"]["max_iter"]
        ), True),
        "Random Forest": (RandomForestModel(
            n_estimators=m_config["models"]["random_forest"]["n_estimators"],
            max_depth=m_config["models"]["random_forest"]["max_depth"]
        ), False),
        "LightGBM": (LightGBMModel(
            n_estimators=m_config["models"]["lightgbm"]["n_estimators"],
            max_depth=m_config["models"]["lightgbm"]["max_depth"],
            learning_rate=m_config["models"]["lightgbm"]["learning_rate"]
        ), False),
        "CatBoost": (CatBoostModel(
            iterations=m_config["models"]["catboost"]["iterations"],
            depth=m_config["models"]["catboost"]["depth"],
            learning_rate=m_config["models"]["catboost"]["learning_rate"]
        ), False),
        "MLP (Neural Net)": (MLPModel(
            hidden_layer_sizes=tuple(m_config["models"]["mlp"]["hidden_layer_sizes"]),
            max_iter=m_config["models"]["mlp"]["max_iter"]
        ), True)
    }

    # 5. Fit and evaluate
    print("\n[Step 3] Fitting baselines and evaluating...")
    results = {}
    best_acc = 0.0
    best_model_name = ""
    
    for name, (model, scaled) in baselines.items():
        tr_x = X_train_scaled if scaled else X_train
        te_x = X_test_scaled if scaled else X_test
        
        t0 = time.time()
        model.fit(tr_x, y_train)
        train_time = time.time() - t0
        
        metrics = model.evaluate(te_x, y_test)
        metrics["train_time_seconds"] = train_time
        
        results[name] = metrics
        
        print(f"  • {name:20s}: Acc: {metrics['accuracy']:.4f} | AUC: {metrics['roc_auc']:.4f} | ECE: {metrics['ece']:.4f} | Fit: {train_time:.2f}s")
        
        if metrics["accuracy"] > best_acc and name not in ["Random Guess", "Majority Predictor"]:
            best_acc = metrics["accuracy"]
            best_model_name = name

    # 6. Statistical Cross-Validation (on the best baseline)
    print(f"\n[Step 4] Running 10-fold CV on Best Baseline Model ({best_model_name})...")
    best_wrapper, scaled = baselines[best_model_name]
    best_model = best_wrapper.model
    
    X_cv = X_train_scaled if scaled else X_train
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    t0 = time.time()
    cv_scores = cross_val_score(best_model, X_cv, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    cv_duration = time.time() - t0
    
    mean_cv = float(np.mean(cv_scores))
    std_cv = float(np.std(cv_scores))
    ci_lower = mean_cv - (1.96 * std_cv)
    ci_upper = mean_cv + (1.96 * std_cv)
    
    cv_stats = {
        "mean_accuracy": mean_cv,
        "std_dev": std_cv,
        "confidence_interval": [ci_lower, ci_upper],
        "duration_seconds": cv_duration
    }
    print(f"    - {best_model_name} 10-fold CV : {mean_cv*100:.2f}% +/- {std_cv*100:.2f}% (95% CI: [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%])")

    # 7. Failure Analysis Slices (on the best baseline)
    print("\n[Step 5] Performing Failure Analysis Slices...")
    best_preds = best_wrapper.predict(X_test_scaled if scaled else X_test)
    correct_mask = (best_preds == y_test)
    
    # Extract trophy difference slice
    # Column 244 is trophy difference (in X_test)
    trophy_diffs = X_test[:, -1]
    
    high_diff = np.abs(trophy_diffs) > 500
    low_diff = np.abs(trophy_diffs) <= 500
    
    high_diff_acc = float(np.mean(correct_mask[high_diff])) if np.sum(high_diff) > 0 else 0.0
    low_diff_acc = float(np.mean(correct_mask[low_diff])) if np.sum(low_diff) > 0 else 0.0
    
    failure_slices = {
        "trophy_diff_gt_500_accuracy": high_diff_acc,
        "trophy_diff_le_500_accuracy": low_diff_acc,
        "trophy_diff_gt_500_samples": int(np.sum(high_diff)),
        "trophy_diff_le_500_samples": int(np.sum(low_diff))
    }
    print(f"  • Slice |Trophy Diff| > 500: Accuracy: {high_diff_acc*100:.2f}% ({np.sum(high_diff)} samples)")
    print(f"  • Slice |Trophy Diff| <= 500: Accuracy: {low_diff_acc*100:.2f}% ({np.sum(low_diff)} samples)")

    # 8. Save Consolidated Report
    report = {
        "model_results": results,
        "best_model": best_model_name,
        "cross_validation": cv_stats,
        "failure_slices": failure_slices
    }
    
    report_path = PROJECT_ROOT / "logs" / "baselines_experiments.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"\n[+] Sprint 11 baselines run completed! Report locked in: {report_path}\n")

if __name__ == "__main__":
    run_baselines_benchmark()
