"""
train_xgboost.py
Trains an XGBoost classifier with scale_pos_weight correction 
to eliminate the majority-class guessing trap.
"""

import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report

# --- 1. Load your preprocessed, archetype-augmented dataframe -------------
df = pd.read_parquet("matches_with_archetypes.parquet")

y = df["win"]

# Dropping categorical tracking strings AND the raw database text deck string blocks
columns_to_drop = ["win", "matchup", "P1_archetype", "P2_archetype", "player_deck", "opponent_deck"]
X = df.drop(columns=columns_to_drop)

print(f"Feeding balanced feature matrix X into XGBoost. Input dimensions: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Calculating exact negative-to-positive class weight ratio: 303 / 403
scale_weight_ratio = 303.0 / 403.0

model = XGBClassifier(
    n_estimators=300,
    max_depth=4,           # keep shallow — with only 3.5k rows, deep trees will overfit
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    scale_pos_weight=scale_weight_ratio,  # <-- CRITICAL FIX: Forces model to learn losses!
    eval_metric="logloss",
    random_state=42,
)

print(f"Fitting XGBoost model (scale_pos_weight={scale_weight_ratio:.4f}) on training partition...")
model.fit(X_train, y_train)
preds = model.predict(X_test)

print("\n=========================================")
print("         HELD-OUT TEST EVALUATION        ")
print("=========================================")
print("Balanced test accuracy:", accuracy_score(y_test, preds))
print("\nClassification Report:")
print(classification_report(y_test, preds))

# --- 2. More trustworthy estimate given the small dataset -----------------
print("\n=========================================")
print("     STRATIFIED 5-FOLD CROSS VALIDATION  ")
print("=========================================")
print("Computing cross-validation folds (this might take a few seconds)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
print(f"5-fold CV accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")

# --- 3. Which features is it actually leaning on? --------------------------
print("\n=========================================")
print("        TOP 20 FEATURE IMPORTANCES       ")
print("=========================================")
importances = pd.Series(model.feature_importances_, index=X.columns)
print(importances.sort_values(ascending=False).head(20))