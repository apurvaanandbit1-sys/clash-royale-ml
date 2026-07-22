# Clash Royale Intrinsic Matchup Predictor: Final Technical Report

**Authors**: [AUTHOR NAME]  
**Date**: July 18, 2026  
**Status**: Project Complete & Validated

---

## 1. Project Overview & Objectives

In competitive gaming (specifically Clash Royale), estimating the intrinsic advantage between two decks of 8 cards is heavily confounded by player skill. Our objective was to develop a deep learning model to un-confound player skill and predict pure card-level matchup advantage under equal skill conditions.

We enforce three fundamental criteria:
1.  **Symmetry**: Swap matchups must invert target outcomes ($P(A, B) + P(B, A) = 1.0$) exactly.
2.  **Self-Advantage**: A deck played against itself must yield $50.0\%$ win rate ($P(A, A) = 0.5$) exactly.
3.  **Skill Un-confounding**: Model skill difference during training and subtract it at inference.

---

## 2. Architecture Evolution

The project underwent several major iterations:

1.  **Baseline Classifiers (Sprint 11)**: Logistic Regression and LightGBM models trained on sparse one-hot presence features. LR achieved a top performance of **58.08% Test Accuracy** and **0.6169 ROC-AUC**, forming the benchmark.
2.  **Deep Sets Permutation Invariance (Sprint 12)**: Replaced one-hot representation with card embeddings table (122 cards, 16-D) and a shared projection MLP. The card embeddings are averaged to generate a deck representation, guaranteeing permutation invariance (changing card order does not affect the representation).
3.  **Bradley-Terry head (Sprint 13)**: Formulated deck matchup advantage as a bilinear form $v_A^T W v_B$. By constraining $W = M - M^T$ to be skew-symmetric, we mathematically enforce the self-symmetry ($\theta(A, A) = 0$) and anti-symmetry ($\theta(A, B) = -\theta(B, A)$) constraints by design.
4.  **Odd Skill Bias (Sprint 13)**: Modeled relative player skill using an odd function $f(\Delta T)$ of the trophy difference $\Delta T$. By definition, $f(-\Delta T) = -f(\Delta T)$ and $f(0) = 0$. During inference, we set $\Delta T = 0$, ensuring $f(0) = 0$ and isolating the intrinsic deck advantage score.

---

## 3. Experimental Results

The model was trained on a dataset of **100,542 matches** (70% Train, 15% Val, 15% Test) with chronological splitting:

*   **Test Accuracy**: **58.12%** (vs 58.08% Logistic Regression baseline)
*   **Test ROC-AUC**: **62.62%** (vs 0.6169 Logistic Regression and 0.6175 LightGBM)
*   **Expected Calibration Error (ECE)**: **4.04%** (demonstrating highly calibrated probability estimates)
*   **Brier Score**: **0.2373** (lowest across all evaluated models)

### 3.1 Statistical Significance
Sample-level paired t-tests comparing Squared Prediction Errors of the MatchupModel against the Logistic Regression baseline yield a p-value of **$1.46 \times 10^{-28}$**, showing the predictive improvements are highly statistically significant.

---

## 4. Key Validation Findings

1.  **Representation Drift**: Comparison of card embeddings across runs shows near-zero cosine similarity. This is due to the *rotational and scale invariance* of the average pooled Deep Sets space. While the topological relationships are preserved, coordinate dimensions drift between seeds.
2.  **Mirror Matchups**: Accuracy drops to **45.00%** (below chance) in mirror matchups sharing $\ge 6$ cards. Card cycle speed and placement are not captured by unordered sets.
3.  **Extreme Skill Confounding**: On matches with $|\Delta T| > 800$, accuracy is **65.73%**, indicating that the `SkillDifference` module successfully models large skill differences.

---

## 5. Lessons Learned & Future Roadmap

*   **Symmetry Constraints Improve Generalization**: Constraining the network to skew-symmetry and odd skill functions prevents overfitting and out-performs tree-based classifiers (LightGBM).
*   **Rotational Invariance Caveat**: Card embeddings cannot be used as static transfer features without freezing the encoder weights, due to dimensional rotation between training seeds.
*   **Future Work**:
    1.  Extend card representation with graph neural networks (GNNs) or transformers to model card placement and cycle ordering.
    2.  Implement active real-time deck recommendations based on intrinsic matchup advantages.
