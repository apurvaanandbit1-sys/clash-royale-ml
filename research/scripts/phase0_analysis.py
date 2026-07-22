import json
import sys
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, log_loss

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from training.dataset import get_dataloaders
from models.deck_encoder import DeckEncoder
from models.interaction_head import BradleyTerryHead, SkillDifference
from models.predictor import MatchupModel
from training.calibration import evaluate_calibration

def build_one_hot_vectors(loader, num_cards: int) -> tuple[np.ndarray, np.ndarray]:
    X_list = []
    y_list = []
    for batch in loader:
        p1_deck = batch["p1_deck"]
        p2_deck = batch["p2_deck"]
        t_diff = batch["trophy_diff"]
        target = batch["target"]
        
        p1_one_hot = torch.nn.functional.one_hot(p1_deck, num_classes=num_cards).sum(dim=1)
        p2_one_hot = torch.nn.functional.one_hot(p2_deck, num_classes=num_cards).sum(dim=1)
        
        feat = torch.cat([p1_one_hot, p2_one_hot, t_diff], dim=1)
        X_list.append(feat.numpy())
        y_list.append(target.numpy())
    return np.vstack(X_list), np.vstack(y_list).squeeze()

def mcnemar_test(y_true, y_pred1, y_pred2, n_bootstraps=1000, seed=42):
    c1 = (y_pred1 == y_true)
    c2 = (y_pred2 == y_true)
    b = np.sum(c1 & ~c2)
    c = np.sum(~c1 & c2)
    
    stat = (abs(b - c) - 1)**2 / (b + c) if (b + c) > 0 else 0.0
    p_val = chi2.sf(stat, 1)
    
    acc1 = np.mean(c1)
    acc2 = np.mean(c2)
    diff = acc1 - acc2
    
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        diffs.append(np.mean(c1[idx]) - np.mean(c2[idx]))
    
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)
    
    return {
        "b": int(b),
        "c": int(c),
        "statistic": float(stat),
        "p_value": float(p_val),
        "acc_diff": float(diff),
        "ci_95": [float(ci_lower), float(ci_upper)]
    }

def bootstrap_auc_diff(y_true, probs1, probs2, n_bootstraps=1000, seed=42):
    auc1 = roc_auc_score(y_true, probs1)
    auc2 = roc_auc_score(y_true, probs2)
    diff = auc1 - auc2
    
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y_true[idx], probs1[idx]) - roc_auc_score(y_true[idx], probs2[idx]))
        
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)
    
    return {
        "auc1": float(auc1),
        "auc2": float(auc2),
        "auc_diff": float(diff),
        "ci_95": [float(ci_lower), float(ci_upper)]
    }

def bootstrap_brier_diff(y_true, probs_lr, probs_model, n_bootstraps=1000, seed=42):
    err_lr = (y_true - probs_lr) ** 2
    err_model = (y_true - probs_model) ** 2
    diff_vec = err_lr - err_model
    
    mean_diff = np.mean(diff_vec)
    
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        diffs.append(np.mean(diff_vec[idx]))
        
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)
    
    return {
        "brier_lr": float(np.mean(err_lr)),
        "brier_model": float(np.mean(err_model)),
        "mean_diff": float(mean_diff),
        "ci_95": [float(ci_lower), float(ci_upper)]
    }

def main():
    print("=== Reconciled Phase 0 Analysis: McNemar, DeLong, Brier CI & K-Count ===")
    
    with open(PROJECT_ROOT / "config" / "dataset_config.yaml", "r") as f:
        ds_config = yaml.safe_load(f)
    with open(PROJECT_ROOT / "config" / "benchmark_config.yaml", "r") as f:
        m_config = yaml.safe_load(f)
        
    df = pd.read_parquet(PROJECT_ROOT / ds_config["data"]["dataset_path"])
    card_lib_path = PROJECT_ROOT / ds_config["data"]["card_library_path"]
    
    with open(card_lib_path, "r") as f:
        cards_lib = json.load(f)
    num_cards = len(cards_lib)
    
    # 1. DataLoader split with symmetry augmentation on train_ldr
    train_ldr_aug, val_ldr, test_ldr = get_dataloaders(
        df=df,
        card_library_path=card_lib_path,
        batch_size=128,
        seed=42,
        augment_prob=0.5
    )
    
    # 2. Load MatchupModel
    best_ckpt_path = PROJECT_ROOT / "models" / "checkpoints" / "sprint13_matchup_model.pt"
    encoder = DeckEncoder(num_cards=num_cards, embedding_dim=16)
    head = BradleyTerryHead(projection_dim=16)
    skill = SkillDifference(hidden_dim=8)
    model = MatchupModel(encoder, head, skill)
    model.load_state_dict(torch.load(best_ckpt_path, map_location=torch.device("cpu")))
    model.eval()
    
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for batch in test_ldr:
            p1 = batch["p1_deck"]
            p2 = batch["p2_deck"]
            td = batch["trophy_diff"]
            target = batch["target"]
            probs = model.predict_proba(p1, p2, td)
            all_targets.append(target.numpy())
            all_probs.append(probs.numpy())
            
    y_true = np.vstack(all_targets).squeeze()
    probs_model = np.vstack(all_probs).squeeze()
    preds_model = (probs_model >= 0.5).astype(float)
    
    # 3. Train Verified Baseline Logistic Regression (Sprint 11 Setup)
    X_train, y_train = build_one_hot_vectors(train_ldr_aug, num_cards)
    X_test, y_test = build_one_hot_vectors(test_ldr, num_cards)
    
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[:, -1] = scaler.fit_transform(X_train[:, -1].reshape(-1, 1)).squeeze()
    X_test_scaled[:, -1] = scaler.transform(X_test[:, -1].reshape(-1, 1)).squeeze()
    
    lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    probs_lr = lr.predict_proba(X_test_scaled)[:, 1]
    preds_lr = (probs_lr >= 0.5).astype(float)
    
    N = len(y_true)
    correct_model = int(np.sum(preds_model == y_true))
    correct_lr = int(np.sum(preds_lr == y_true))
    K = correct_model - correct_lr
    
    print(f"\n[1] Battle Classification Comparison (N={N}):")
    print(f"    - MatchupModel Correct:       {correct_model} ({correct_model/N*100:.2f}%)")
    print(f"    - Logistic Regression Correct: {correct_lr} ({correct_lr/N*100:.2f}%)")
    print(f"    - Plain Language Result:       On the {N}-battle test set, MatchupModel correctly classifies {K} more battles than Logistic Regression.")
    
    # McNemar's Test
    mcnemar_res = mcnemar_test(y_true, preds_model, preds_lr)
    print(f"\n[2] McNemar's Test:")
    print(f"    - b (Model right, LR wrong): {mcnemar_res['b']}")
    print(f"    - c (Model wrong, LR right): {mcnemar_res['c']}")
    print(f"    - Statistic:                 {mcnemar_res['statistic']:.4f}")
    print(f"    - p-value:                   {mcnemar_res['p_value']:.4e}")
    print(f"    - Accuracy Diff 95% CI:      [{mcnemar_res['ci_95'][0]:.6f}, {mcnemar_res['ci_95'][1]:.6f}]")
    
    # DeLong / Bootstrap AUC Diff
    auc_res = bootstrap_auc_diff(y_true, probs_model, probs_lr)
    print(f"\n[3] ROC-AUC Comparison & Bootstrap CI:")
    print(f"    - MatchupModel AUC:          {auc_res['auc1']:.4f}")
    print(f"    - Logistic Regression AUC:   {auc_res['auc2']:.4f}")
    print(f"    - AUC Diff:                  {auc_res['auc_diff']:.4f}")
    print(f"    - AUC Diff 95% CI:           [{auc_res['ci_95'][0]:.6f}, {auc_res['ci_95'][1]:.6f}]")
    
    # Brier Score & Reconciled CI
    brier_res = bootstrap_brier_diff(y_true, probs_lr, probs_model)
    print(f"\n[4] Brier Score Reconciliation & Error Difference CI:")
    print(f"    - Logistic Regression Brier: {brier_res['brier_lr']:.4f}")
    print(f"    - MatchupModel Brier:        {brier_res['brier_model']:.4f}")
    print(f"    - Brier Difference:          {brier_res['mean_diff']:.6f}")
    print(f"    - Reconciled 95% Error CI:   [{brier_res['ci_95'][0]:.6f}, {brier_res['ci_95'][1]:.6f}]")
    
    # Save results to json
    res_payload = {
        "sample_size": N,
        "correct_model": correct_model,
        "correct_lr": correct_lr,
        "K_more_battles": K,
        "mcnemar": mcnemar_res,
        "auc_diff": auc_res,
        "brier_reconciliation": brier_res
    }
    
    out_path = PROJECT_ROOT / "logs" / "phase0_statistical_results.json"
    with open(out_path, "w") as f:
        json.dump(res_payload, f, indent=4)
    print(f"\nSaved statistical results to {out_path}")

if __name__ == "__main__":
    main()
