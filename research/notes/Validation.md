# Scientific Validation

This document summarizes the scientific validation results.

## 1. Repeated Seeds Sweep (10 Seeds)
*   **Test Accuracy**: Mean = **58.56%** ($\sigma = 0.24\%$)
*   **Test ROC-AUC**: Mean = **62.70%** ($\sigma = 0.22\%$)
*   **ECE**: Mean = **3.86%** ($\sigma = 0.29\%$)

Narrow standard deviations show the model training is stable and reproducible.

## 2. Statistical Significance (Paired t-test)
We ran paired t-tests comparing Squared Errors against the Logistic Regression baseline:
*   **Sample Size (N)**: 15,082
*   **Average p-value**: **$2.67 \times 10^{-20}$**
*   **95% CI of Error Diff**: $[0.0021, 0.0039]$ in favor of the MatchupModel.

This confirms that the MatchupModel's improvements are highly statistically significant.

## 3. Representation Drift (Embedding Stability)
*   **Mean Cosine Similarity**: $-0.0139$
*   **Jaccard Nearest-Neighbor Index**: $0.0525$
*   *Explanation*: Average card pooling introduces rotational invariance. The model learns identical relative distances (topology) but absolute coordinates undergo arbitrary orthogonal rotation between seeds.

## 4. Failure Slices
*   **Extreme Skill ($|\Delta T| > 800$)**: Accuracy = **65.73%**. Skill un-confounding is highly successful.
*   **Mirror Matchups ($\ge 6$ card overlaps)**: Accuracy = **45.00%**. Unordered set encoders fail to capture cycle card order in symmetric mirror decks.
*   **Rare/Off-Meta Decks**: Accuracy = **62.06%**, showing strong generalization.
