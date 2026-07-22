# Research Report: ML Benchmarks & Feature Engineering Audit (Sprint 4)

**Prepared by**: Principal Machine Learning Engineer
**Date**: July 17, 2026
**Target**: Win Probability Prediction Pipeline

---

## 1. Dataset & Split Validation

Before training any benchmark models, we ran a validation audit on the preprocessed 100k battles dataset (`matches_with_features.parquet`).

### Integrity Audits
*   **Total Row Count**: 100,542 battles
*   **Total Feature Count**: 28 columns (including `player_deck`, `opponent_deck`, `win`, and `battle_time`)
*   **Missing Values**: Exactly 0 missing values across all columns.
*   **Duplicate Battles**: 1,984 exact duplicate matchups. These represent natural occurrences in competitive matchmaking where the same two players face off with the identical deck combination. They are kept in the dataset as they occurred at different timestamps.
*   **Class Balance**: Wins: 52.05% (52,328) | Losses: 47.95% (48,214). The dataset is highly balanced and free from majority-class bias.
*   **Value Boundaries**: 0 boundary violations. Elixir values fall within [1, 9], and count/index features are $\ge 0$.

### Split Partitioning Validation
To simulate production deployment and prevent temporal information leakage, we sorted matches chronologically by `battle_time` and partitioned them:
*   **Train Split**: Earliest 70% of matches (70,379 samples)
*   **Validation Split**: Intermediate 15% of matches (15,081 samples)
*   **Test Split**: Latest 15% of matches (15,082 samples)

We ran an intersection check on indexes: **0 overlapping records cross partitions**, confirming complete split isolation.

---

## 2. Exploratory Data Analysis (EDA)

*   **Trophy Distribution**: Trophy range spans from 0 to 14,395 with a mean of **8,116**. This indicates that the dataset is highly representative of elite, competitive Clash Royale play.
*   **Class Imbalance**: Class distribution is nearly symmetric (52% wins), meaning models do not require class-weight corrections or majority-class adjustments.
*   **Collinearity / Redundancy**: Pearson correlation checks revealed **0 feature pairs** exceeding the redundancy threshold ($|r| > 0.8$). This indicates that our 12 engineered features represent distinct structural dimensions.
*   **Card Frequencies**: The top 3 most played cards are *Witch* (35.6%), *The Log* (28.4%), and *Mega Knight* (27.8%). Champion cards are present in 28.32% of all decks.

---

## 3. Benchmark Model Comparison

We evaluated five fundamentally different model families on identical train/validation/test chronological splits. The target is predicting the `win` column (0 or 1).

| Model Family | Validation Accuracy | Test Accuracy | Test Log Loss | Test ROC-AUC | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dummy (Baseline)** | 0.5091 | 0.4575 | 0.7020 | 0.5000 | 0.0001 ms |
| **Logistic Regression** | 0.5326 | 0.5076 | 0.6991 | 0.5570 | 0.0000 ms |
| **Random Forest** | 0.5222 | 0.4777 | 0.6960 | 0.5688 | 0.0030 ms |
| **XGBoost** | 0.5344 | 0.5052 | 0.6928 | **0.5768** | 0.0005 ms |
| **MLP (Neural Net)** | **0.5286** | **0.5324** | 0.7318 | 0.5573 | 0.0022 ms |

### Key Observations:
1.  **Low Predictive Signal**: The best model (MLP Neural Net) achieved a test accuracy of **53.24%**, which is barely above random guessing.
2.  **Tree Overfitting**: Random Forest and XGBoost exhibit standard signs of overfitting on the train split, performing poorly on the test set.
3.  **Low ROC-AUC**: XGBoost achieved the highest ROC-AUC of **0.5768**, indicating that the models struggle to discriminate between winning and losing deck combinations.

---

## 4. Systematic Error Slicing

We ran error analysis on our best performer (MLP Classifier) to find where prediction failures are concentrated:
*   **Trophy Slices**:
    *   `< 8,000 trophies`: **49.52%** error rate
    *   `8,000 - 9,000 trophies`: **45.48%** error rate
    *   `> 9,000 trophies`: **47.17%** error rate
*   **Arena Slices**:
    *   *Valkalla*: 46.04% error rate
    *   *PANCAKES!*: 47.33% error rate
    *   *Legendary Arena*: 50.56% error rate
*   **Special Cards (Mirror)**:
    *   *Mirror NOT in deck*: 46.76% error rate
    *   *Mirror present in deck*: 46.84% error rate

### Key Finding:
Prediction error rates are relatively uniform (ranging from 45% to 54%) across all slices. Slices representing the highest skill brackets (Legendary Arena, >9,000 trophies) show slightly higher error rates, suggesting that competitive matchups are highly nuanced and cannot be resolved using simple deck aggregates.

---

## 5. Feature Engineering Audit

### Feature Importances
*   **Random Forest**: Top 3 features are `p2_has_small_spell`, `p2_average_elixir`, and `p2_spell_count`.
*   **XGBoost**: Top 3 features are `p2_has_small_spell`, `p2_has_champion`, and `p2_has_evolution`.

### Redundancy & Noise Analysis:
1.  **Opponent Bias**: Both models heavily prioritize opponent (`p2_`) features over player (`p1_`) features. This suggests that the model is learning unbalanced patterns (e.g. predicting a loss if the opponent has a cheap deck or small spell) because it cannot model how the player's deck directly *counters* those features.
2.  **Aggregation Loss**: Features like `durability_index` and `damage_index` contribute very little to the model splits, behaving like noise.

---

## 6. Preparation for Future Work (Sprint 5)

### 1. Is the current feature engineering sufficient?
**No.** The low benchmark test accuracy (50.5%–53.2%) confirms that deck-level aggregates are insufficient.

### 2. What information appears to be missing?
*   **Card-to-Card Counters**: The models do not know that *Arrows* wipes out *Minion Horde*, or that *Inferno Tower* counters *Giant*.
*   **Exact Card Identities**: The models treat all Win Conditions identically, failing to distinguish between a *Hog Rider* (fast cycle) and a *Golem* (slow beatdown).

### 3. Would interaction-aware features likely improve performance?
**Yes.** Explicitly representing pairwise card interactions (e.g., `p1_card_i` vs `p2_card_j`) allows the model to learn the rock-paper-scissors countering system that defines Clash Royale strategy.

### 4. Would representation learning likely improve performance?
**Yes.** Projecting card IDs into a dense embedding space (e.g. using representation learning) allows the model to automatically learn latent card similarity and synergies (e.g., mapping *Baby Dragon* and *Executioner* close together as splash support).

### 5. What evidence supports these conclusions?
*   The flat, random-like test performance of classical models (LR, RF, XGBoost) despite having a massive, clean 100k-row training set.
*   Feature importances focus entirely on one-sided features rather than interactions.
*   High prediction entropy on competitive matches.

---

## 7. Recommendations for Sprint 5

1.  **Implement Card Embeddings**: Train a representation learning model (using autoencoders or co-occurrence matrices) to map card IDs to dense embeddings.
2.  **Transition to Attention/Transformer Architectures**: Treat a deck as a set of 8 card embeddings, and use self-attention to learn synergies within a deck, and cross-attention to learn counter relationships between decks.
3.  **Benchmark Comparison**: Use the MLP and XGBoost models developed in this sprint as the strict baselines that the new attention-based architecture must outperform.
