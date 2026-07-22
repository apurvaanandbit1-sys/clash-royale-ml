# Embedding Architecture Research & Design Blueprint (Sprint 7)

**Prepared by**: Principal Machine Learning Research Scientist  
**Date**: July 17, 2026  
**Target**: Learned Card Representations for Win Probability Prediction

---

## 1. Review of Experimental Evidence

### What Has Been Experimentally Validated:
1.  **Card Identity is Essential**: Sprint 6 proved that preserving explicit card identities (Rep B) yields a statistically significant increase of **+5.82% in test accuracy** (from 50.76% to 56.58%) and **+0.053 in ROC-AUC** over aggregate deck statistics (Rep A).
2.  **Aggregates act as Redundancy/Noise**: Combining aggregates with one-hot indicators (Rep C) failed to outperform one-hot indicators alone, indicating that manual features provide no additional value once card identities are available.
3.  **High Dimensionality Overfits**: Training a Multi-Layer Perceptron (MLP) on 244-dimensional sparse one-hot binary inputs led to severe overfitting, raising the Expected Calibration Error (ECE) to **0.3406** on the test partition.
4.  **Intrinsic Information Ceiling**: Predictions from static deck lists alone have an estimated ceiling of **60%–65% accuracy** due to hidden variables like card placements, starting hands, rotations, and player skill.

### Rejected Hypotheses:
*   *Hypothesis A (Aggregates contain most signal)*: Rejected.
*   *Hypothesis C (Identity loss is low so aggregates are sufficient)*: Rejected. Although deck aggregates rarely collide (1.69% rate), the models cannot extract counter dynamics from sum statistics.

---

## 2. Representation Learning Survey

We evaluated several embedding methodologies to map the $C \approx 120$ card IDs to a dense vector space:

| Approach | Advantages | Disadvantages | Expected Sample Efficiency | Ease of Implementation | Suitability for Clash Royale |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Random Trainable Lookup** | Simple to code; optimized directly for win prediction task. | High risk of overfitting on rare cards; no initial semantic structure. | Low (requires many training iterations) | High | **Medium** |
| **Autoencoder Compression** | Learns unsupervised deck-level distributions. | Does not optimize for prediction outcome; complex two-stage training. | Medium | Medium | **Low** |
| **SVD Co-occurrence (Decks as Sentences)** | Captures semantic similarities and meta card-synergies from 100k battles. | Does not incorporate win outcomes during initialization. | High (pre-trained offline on deck lists) | High | **High (Recommended)** |
| **Contrastive Metric Learning** | Directly pulls winning decks close and pushes losing decks apart. | Requires careful negative sampling; training instability. | Medium | Low | **Medium** |
| **Meta-Informed Initialization** | Initializes cards using attributes (elixir, rarity, role) from knowledge base. | Biased toward manual attributes; may ignore latent playstyle synergies. | High | Medium | **High (Combined)** |

---

## 3. Recommended Embedding Design

We recommend a **Hybrid SVD Co-occurrence Pre-training + End-to-End Fine-tuning** embedding architecture.

```
       [Raw Deck List] -> [SVD Card Co-occurrence Matrix (Pre-trained)]
                                       |
                                       v
                             [Embedding Lookup Table] (16-Dim)
                                       |
                                       v
                            [End-to-End Fine-tuning] (MLP/Attention)
```

### 1. Dimensionality ($d = 16$)
*   **Justification**: Applying the standard rule of thumb $d \approx 4 \times \sqrt[4]{C}$ on $C = 122$ cards yields $d \approx 13.2$. Setting $d = 16$ (a power of 2) balances network capacity and regularization, reducing input features from 244 sparse binary dimensions to 32 dense continuous dimensions per match (after set pooling).

### 2. Pre-Training Strategy (SVD Co-occurrence)
*   **Method**: Build a $C \times C$ co-occurrence matrix $W$ from all winning decks in the 100k database. If Card $i$ and Card $j$ frequently appear together in winning decks, they share high synergy (e.g. Giant + Sparky).
*   **Decomposition**: Perform Singular Value Decomposition (SVD) on the normalized co-occurrence matrix:
    $$W \approx U \Sigma V^T$$
    Select the top 16 singular vectors from $U$ to serve as the initial weights for our embedding lookup table. This guarantees that cards with similar roles are placed close together in the latent space before prediction training begins.

### 3. Handling Game Updates & Exceptions
*   **Balance Changes**: Card stats (HP, damage) change frequently in game patches. Because our embedding maps functional synergy (how cards are played together) rather than hardcoded stats, balance changes will naturally update the co-occurrence matrix over time during standard crawling.
*   **Evolutions & Champions**: Treat evolved cards as distinct states. For example, Card ID `26000000` (Knight) and Card ID `26100000` (Evolved Knight) will have separate rows in the embedding lookup table, allowing the model to learn the distinct power multiplier of Evolutions.

---

## 4. Downstream Model Integration

We compared how the 16-dimensional card embeddings will integrate with prediction models:

1.  **Embedding Bag (Deep Sets)**:
    *   *Method*: Retrieve the eight 16-dimensional embeddings for a deck and sum or average pool them into a single 16-dimensional deck vector. Concatenate Player 1 and Player 2 vectors and pass to an MLP.
    *   *Pros/Cons*: Permutation invariant, simple, and parameters are independent of card order. Highly sample-efficient.
    *   *Verdict*: **Evaluate first (Baseline)**.
2.  **Cross-Attention Interaction Net**:
    *   *Method*: Project Player 1's 8 embeddings as Queries ($Q$) and Player 2's 8 embeddings as Keys ($K$) and Values ($V$). Multi-head attention directly computes pairwise card interactions.
    *   *Pros/Cons*: Represents rock-paper-scissors countering relationships directly; maps explicit card-to-card counters. Higher parameter count.
    *   *Verdict*: **Evaluate second (Primary Target)**.

---

## 5. Experimental Evaluation Plan

*   **Baseline**: Sprints 6 One-Hot Logistic Regression (56.58% accuracy).
*   **Success Criteria**:
    1.  *Performance*: Test accuracy exceeding **58.5%** (crossing the upper bound of the Sprints 6 95% Confidence Interval).
    2.  *Calibration*: Brier score $< 0.2400$, and Expected Calibration Error (ECE) $< 0.05$ (indicating well-calibrated win probabilities).
*   **Ablation Runs**:
    *   *Ablation 1*: Random Initialization vs. SVD Co-occurrence Pre-training.
    *   *Ablation 2*: Trainable Embeddings vs. Frozen Embeddings.

---

## 6. Risks & Mitigation Strategies

1.  **Rare/Sparse Cards**: Cards with low play rates will have noisy gradients.
    *   *Mitigation*: Apply **weight decay (L2 regularization)** specifically to the embedding layer to pull unplayed card vectors toward the origin.
2.  **Embedding Collapse**: Embeddings mapping to a single point during training.
    *   *Mitigation*: Apply **dropout** to the embedding layer (randomly zeroing out vector dimensions) to force the network to distribute features across all 16 dimensions.
3.  **Patch Drift**: Metas shift after balance changes, rendering old embeddings obsolete.
    *   *Mitigation*: Implement a weekly cron job to update the co-occurrence matrix from the latest 10,000 battles and warm-start the lookup table.
