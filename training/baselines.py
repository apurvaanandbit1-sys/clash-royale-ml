import os
import pickle
import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score, precision_score, recall_score
from training.calibration import evaluate_calibration

class CRBaselineModel:
    """
    Unified baseline model interface.
    Every baseline model must implement fit, predict, predict_proba, save, and load.
    """
    def __init__(self):
        self.model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Computes all required validation metrics."""
        t0 = time.time()
        preds = self.predict(X_test)
        inference_time_total = time.time() - t0
        latency_ms = (inference_time_total / len(X_test)) * 1000

        probs = self.predict_proba(X_test)
        cal = evaluate_calibration(y_test, probs)

        metrics = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall": float(recall_score(y_test, preds, zero_division=0)),
            "f1_score": float(f1_score(y_test, preds, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probs)),
            "log_loss": float(log_loss(y_test, probs, labels=[0, 1])),
            "brier_score": float(cal["brier_score"]),
            "ece": float(cal["ece"]),
            "inference_latency_ms": latency_ms
        }
        return metrics

    def save(self, filepath: str):
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, filepath: str):
        with open(filepath, "rb") as f:
            self.model = pickle.load(f)


class RandomGuessModel(CRBaselineModel):
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.random.choice([0, 1], size=len(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.ones(len(X)) * 0.5

    def save(self, filepath: str):
        with open(filepath, "w") as f:
            f.write("RandomGuessModel")

    def load(self, filepath: str):
        pass


class MajorityClassModel(CRBaselineModel):
    def __init__(self):
        super().__init__()
        self.majority_class = 1.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        counts = np.bincount(y_train.astype(int))
        self.majority_class = float(np.argmax(counts))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.ones(len(X)) * self.majority_class

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.ones(len(X)) * self.majority_class

    def save(self, filepath: str):
        with open(filepath, "wb") as f:
            pickle.dump(self.majority_class, f)

    def load(self, filepath: str):
        with open(filepath, "rb") as f:
            self.majority_class = pickle.load(f)


class LogisticRegressionModel(CRBaselineModel):
    def __init__(self, C: float = 0.1, max_iter: int = 1000, random_seed: int = 42):
        super().__init__()
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression(C=C, max_iter=max_iter, random_state=random_seed)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


class RandomForestModel(CRBaselineModel):
    def __init__(self, n_estimators: int = 100, max_depth: int = 6, random_seed: int = 42):
        super().__init__()
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_seed, n_jobs=-1)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


class LightGBMModel(CRBaselineModel):
    def __init__(self, n_estimators: int = 100, max_depth: int = 4, learning_rate: float = 0.05, random_seed: int = 42):
        super().__init__()
        from lightgbm import LGBMClassifier
        self.model = LGBMClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            learning_rate=learning_rate, 
            random_state=random_seed, 
            n_jobs=-1,
            verbose=-1
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


class CatBoostModel(CRBaselineModel):
    def __init__(self, iterations: int = 100, depth: int = 4, learning_rate: float = 0.05, random_seed: int = 42):
        super().__init__()
        from catboost import CatBoostClassifier
        self.model = CatBoostClassifier(
            iterations=iterations, 
            depth=depth, 
            learning_rate=learning_rate, 
            random_seed=random_seed, 
            verbose=0,
            thread_count=-1
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


class MLPModel(CRBaselineModel):
    def __init__(self, hidden_layer_sizes: tuple = (64, 32), max_iter: int = 300, random_seed: int = 42):
        super().__init__()
        from sklearn.neural_network import MLPClassifier
        self.model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=random_seed)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]
