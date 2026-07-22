import time
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from training.experiment_tracker import log_experiment

def train_and_evaluate_benchmarks(
    X_train: pd.DataFrame, 
    X_val: pd.DataFrame, 
    X_test: pd.DataFrame, 
    y_train: pd.Series, 
    y_val: pd.Series, 
    y_test: pd.Series,
    dataset_version: str,
    random_seed: int = 42
) -> dict:
    """
    Trains and evaluates standard benchmark models on the identical splits.
    Models evaluated:
        1. Dummy Classifier (Baseline)
        2. Logistic Regression (L2 Regularized)
        3. Random Forest Classifier
        4. XGBoost Classifier
        5. Multi-Layer Perceptron (MLP Neural Net)
    """
    
    # Standardize/Normalize inputs for models that require scaling (Logistic Regression & MLP)
    # Binary/boolean features should remain as 0/1. Let's do a simple scaling for safety.
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    # Scale only continuous features, leave binary features untouched to preserve interpretability,
    # or scale everything. For MLP/Logistic, scaling everything is standard.
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert booleans in raw X to numeric (0/1) for XGBoost and Random Forest
    X_train_num = X_train.astype(float)
    X_val_num = X_val.astype(float)
    X_test_num = X_test.astype(float)

    models_config = {
        "Dummy (Baseline)": {
            "model": DummyClassifier(strategy="prior"),
            "scaled": False,
            "params": {"strategy": "prior"}
        },
        "Logistic Regression": {
            "model": LogisticRegression(C=0.1, max_iter=1000, random_state=random_seed),
            "scaled": True,
            "params": {"C": 0.1, "penalty": "l2"}
        },
        "Random Forest": {
            "model": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=random_seed, n_jobs=-1),
            "scaled": False,
            "params": {"n_estimators": 100, "max_depth": 6}
        },
        "XGBoost": {
            "model": XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, eval_metric="logloss", random_state=random_seed, n_jobs=-1),
            "scaled": False,
            "params": {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.05}
        },
        "MLP (Neural Net)": {
            "model": MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", max_iter=500, random_state=random_seed),
            "scaled": True,
            "params": {"hidden_layer_sizes": (64, 32), "activation": "relu"}
        }
    }

    results = {}

    for name, config in models_config.items():
        print(f"\nTraining model: {name}...")
        
        # Select appropriate inputs
        if config["scaled"]:
            train_x, val_x, test_x = X_train_scaled, X_val_scaled, X_test_scaled
        else:
            train_x, val_x, test_x = X_train_num, X_val_num, X_test_num
            
        model = config["model"]
        
        # Measure training time
        t0 = time.time()
        model.fit(train_x, y_train)
        train_time = time.time() - t0
        
        # Evaluate on Val & Test sets
        # Measure inference time per sample on test set
        t1 = time.time()
        preds = model.predict(test_x)
        inference_time_total = time.time() - t1
        inference_latency = inference_time_total / len(test_x)
        
        # Probabilities for logloss and ROC-AUC
        if hasattr(model, "predict_proba"):
            val_probs = model.predict_proba(val_x)[:, 1]
            test_probs = model.predict_proba(test_x)[:, 1]
        else:
            val_probs = preds
            test_probs = preds

        # Metrics computation
        metrics = {
            "val_accuracy": float(accuracy_score(y_val, model.predict(val_x))),
            "val_logloss": float(log_loss(y_val, val_probs, labels=[0, 1])),
            "val_roc_auc": float(roc_auc_score(y_val, val_probs)),
            "val_f1_score": float(f1_score(y_val, model.predict(val_x), average="binary")),
            
            "test_accuracy": float(accuracy_score(y_test, preds)),
            "test_logloss": float(log_loss(y_test, test_probs, labels=[0, 1])),
            "test_roc_auc": float(roc_auc_score(y_test, test_probs)),
            "test_f1_score": float(f1_score(y_test, preds, average="binary"))
        }

        print(f"Results for {name}:")
        print(f"  • Val Acc:   {metrics['val_accuracy']:.4f} | Test Acc:   {metrics['test_accuracy']:.4f}")
        print(f"  • Val LogL:  {metrics['val_logloss']:.4f} | Test LogL:  {metrics['test_logloss']:.4f}")
        print(f"  • Val AUC:   {metrics['val_roc_auc']:.4f} | Test AUC:   {metrics['test_roc_auc']:.4f}")
        print(f"  • Fit Time:  {train_time:.2f}s | Latency: {inference_latency*1000:.4f}ms/sample")

        # Save to logs/experiments.json
        log_experiment(
            model_name=name,
            params=config["params"],
            metrics=metrics,
            train_time=train_time,
            inference_time=inference_latency,
            dataset_version=dataset_version,
            random_seed=random_seed
        )

        results[name] = {
            "model": model,
            "metrics": metrics,
            "train_time": train_time,
            "inference_latency": inference_latency
        }

    return results
