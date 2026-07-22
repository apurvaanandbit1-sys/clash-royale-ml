# Technical Design Document: Causal Matchup Formulation & Pairwise Bradley-Terry Architecture (Sprint 8)

**Prepared by**: Principal Machine Learning Research Scientist  
**Date**: July 17, 2026  
**Target**: Re-Formulating the Prediction Objective & Architectural Migration

---

## 1. Evaluation of the Current Objective

The current machine learning formulation defines the prediction objective as:
$$\hat{y} = P(Y = 1 \mid D_1, D_2)$$
Where $Y \in \{0, 1\}$ represents the observed match outcome (1 if Player 1 wins, 0 if Player 2 wins), and $D_1, D_2$ represent Player 1 and Player 2 deck feature matrices.

### Scientific and Statistical Limitations:
1.  **Confounding via Player Skill**: The observed outcome $Y$ is a function of both the *intrinsic deck matchup advantage* ($M_{1,2}$) and the *relative player skill* ($S_1 - S_2$). By ignoring player skill during training, the model forces the deck representations ($D_1, D_2$) to absorb the variance of player skill. This introduces massive **omitted variable bias**.
2.  **Product Alignment Mismatch**: Our goal is to determine the intrinsic matchup advantage (e.g. "If two equally skilled players repeatedly played these decks, what percentage of games would each deck win?"). Under the current formulation, if a highly skilled player wins with a weak deck, the model incorrectly attributes predictive weight to the deck features, leading to false indicators in card synergies.
3.  **High-Entropy Label Noise**: Game-level outcomes in ladder matches contain massive noise (starting hand luck, connection latency, misplays). A binary win label on a single match is a highly noisy proxy for the true matchup advantage.

---

## 2. Formulation of the Correct Learning Target

We propose that the model should learn the **Intrinsic Deck Matchup Advantage**:
$$A(D_a, D_b) = P(Y = 1 \mid D_a, D_b, \Delta S = 0)$$
Where $\Delta S = S_a - S_b$ represents the difference in player skill, explicitly set to $0$.

### Mathematical Justification:
Let the observed outcome $Y$ be governed by a latent difference in score $z$:
$$z = \theta(D_a, D_b) + \gamma(S_a, S_b) + \epsilon$$
Where:
*   $\theta(D_a, D_b)$ is the intrinsic deck advantage function.
*   $\gamma(S_a, S_b)$ is the relative player skill function.
*   $\epsilon \sim \text{Logistic}(0, 1)$ represents random in-game noise.

The probability of Player A winning is given by the sigmoid link function:
$$P(Y = 1 \mid D_a, D_b, S_a, S_b) = \sigma(\theta(D_a, D_b) + \gamma(S_a, S_b))$$
At inference time, to evaluate the pure deck matchup, we perform a **counterfactual intervention** setting $S_a = S_b$ (hence $\gamma(S_a, S_b) = 0$):
$$P(Y = 1 \mid D_a, D_b, \text{do}(\Delta S = 0)) = \sigma(\theta(D_a, D_b))$$

---

## 3. Causal Confounding Analysis

We represent the system using a **Directed Acyclic Graph (DAG)**:

```
    [Player 1 Skill (S1)]       [Player 2 Skill (S2)]
             \                       /
              v                     v
  [Deck 1 (D1)] -> [Outcome (Y)] <- [Deck 2 (D2)]
```

*   **The Confounding Mechanism**: Player skill ($S$) is a confounder because high-skill players are more likely to select specific high-synergy meta decks ($D$). Consequently, the deck selection $D$ is causally dependent on skill $S$. If we do not adjust for $S$, the causal path $D \rightarrow Y$ is confounded by the back-door path $D \leftarrow S \rightarrow Y$.
*   **Trophies as a Skill Proxy**: Trophy counts ($T_1, T_2$) are a noisy but valid proxy for player skill ($S$). Therefore, we can model skill using a parameterized function of trophies:
    $$S_i = f(T_i)$$
*   **The "Training vs. Inference" Rule**:
    *   **Training**: Player trophies $T_1$ and $T_2$ **must** remain in the feature set during training to close the backdoor path and isolate the deck effect $\theta(D_a, D_b)$.
    *   **Inference**: Player trophies $T_1$ and $T_2$ **must** be set to equal values (e.g. $T_1 = T_2 = 8000$) to evaluate the matchup under equal skill, isolating the pure deck interaction.

---

## 4. Architectural Redesign

We recommend a **Siamese Deep Sets Bradley-Terry Architecture**:

```
[Deck A (IDs)] ----> [Card Embeddings (16-D)] ----> [Deep Sets Sum Pool] ----> [Deck Latent Vector v_A (16-D)] --\
                                                                                                                   \
                                                                                                                    v
                                                                                                               [Bilinear Interaction] ---> \theta(D_A, D_B)
                                                                                                                    ^
                                                                                                                   /
[Deck B (IDs)] ----> [Card Embeddings (16-D)] ----> [Deep Sets Sum Pool] ----> [Deck Latent Vector v_B (16-D)] --/
                                                                                                                   |
[Trophies T_A] ----------------------------------------------------------------------------------------------------> [Skill Diff Net] -------> \gamma(T_A, T_B)
                                                                                                                                                   |
                                                                                                                                                   v
                                                                                                                                       [Sigmoid Activation] ---> P(A wins)
```

### Component Details:
1.  **Shared Deck Encoder**: A Siamese network that encodes Deck A and Deck B into dense 16-dimensional vectors $v_A, v_B$ using a shared lookup table and Deep Sets sum pooling.
2.  **Bilinear Interaction Head**: Computes the matchup score:
    $$\theta(D_A, D_B) = v_A^T W v_B$$
    Where $W \in \mathbb{R}^{16 \times 16}$ is a learnable antisymmetric weight matrix (to enforce mathematical symmetry, $W^T = -W$).
3.  **Skill Difference Network**: Maps player trophies $T_A, T_B$ to relative skill:
    $$\gamma(T_A, T_B) = w_s \cdot (T_A - T_B)$$
4.  **Bradley-Terry Logit Fusion**: Combines scores into the sigmoid function:
    $$P(\text{A wins}) = \sigma(v_A^T W v_B + w_s(T_A - T_B))$$

---

## 5. Mathematical Constraints

To prevent non-physical predictions, the architecture enforces these constraints **by design**:
1.  **Identity Symmetry**: A deck playing against itself must have exactly 50% win probability:
    $$P(A \text{ vs } A) = 50\%$$
    *Proof*: Since $W$ is antisymmetric ($W^T = -W$), the bilinear product is zero:
    $$v_A^T W v_A = 0 \implies \sigma(0) = 0.50 \text{ (or 50\%)}$$
2.  **Anti-Symmetry**: The win probability of B against A must be the complement of A against B:
    $$P(A \text{ beats } B) + P(B \text{ beats } A) = 100\%$$
    *Proof*: Since $v_B^T W v_A = (v_B^T W v_A)^T = v_A^T W^T v_B = -v_A^T W v_B$, the logits are exactly negated, guaranteeing:
    $$\sigma(x) + \sigma(-x) = 1.0$$

These constraints are guaranteed by the antisymmetric weight matrix $W$, bypassing the need for post-hoc calibration.

---

## 6. Training Loss Formulation

We recommend the **Bradley-Terry Likelihood Loss** (which corresponds to Binary Cross-Entropy on the symmetric logit difference):
$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]$$
Where:
$$\hat{p}_i = \sigma(v_{A,i}^T W v_{B,i} + w_s(T_{A,i} - T_{B,i}))$$

### Comparison of Loss Functions:
*   **Bradley-Terry BCE (Recommended)**: Learns both absolute scale and calibrated probabilities; directly fits the sigmoid output.
*   **Pairwise Margin Ranking Loss**: Only learns the ordinal ranking of matchups, failing to output calibrated probability percentages (e.g. 63% win rate).

---

## 7. Inference API Design

The production API will accept only the two deck card lists and return the intrinsic matchup advantage.

### Python API Interface:
```python
def predict_matchup_advantage(deck_a: list[int], deck_b: list[int]) -> float:
    """
    Predicts the intrinsic win probability of Deck A against Deck B.
    Accepts ONLY card lists; player metadata is fixed to 0.
    """
    # 1. Encode card lists into embeddings
    v_a = shared_deck_encoder(deck_a)
    v_b = shared_deck_encoder(deck_b)
    
    # 2. Compute bilinear matchup score (with S_diff = 0)
    theta = float(v_a.T @ W @ v_b)
    
    # 3. Apply sigmoid activation
    win_probability = 1.0 / (1.0 + np.exp(-theta))
    return win_probability
```

---

## 8. Validation Strategy

To verify that the model measures *matchup advantage* rather than ladder noise:
1.  **Mirror Match Sanity Check**: Evaluate 1,000 random deck pairs under $A(D_i, D_i)$. The predicted win probability must be **exactly 50.00%** (mathematical sanity check).
2.  **Hard Counter Test Set**: Compile a small evaluation set of 100 historical hard-counter deck matchups (e.g. Log Bait vs. Giant Beatdown). Verify that the model outputs matchup probabilities $> 60\%$ in favor of the counter deck.
3.  **Trophy Ablation Study**: Compare models trained with the trophy difference adjustment against models trained without it. The model with trophy adjustment should achieve lower validation entropy on high-trophy competitive subsets.

---

## 9. Migration Roadmap

1.  **Component Preservation**:
    *   Preserve the database schema and parquet generation files.
    *   Preserve the card library data dictionary.
2.  **Component Replacement**:
    *   Modify the preprocessor to load the `player_trophies` and `opponent_trophies` columns from SQLite and save them in the Parquet file.
    *   Replace the flat classifier training scripts in `training/` with a PyTorch training pipeline implementing the Siamese Bradley-Terry model.
3.  **Complexity Estimation**:
    *   *Data updates*: Low (1 hour to update SQL queries in preprocessor).
    *   *Model implementation*: Medium (1–2 days to implement PyTorch bilinear Siamese training).

---

## 10. Critical Review & Defense

### Possible Weaknesses:
*   *Assumption of Linear Trophy-Skill Relationship*: Trophies do not scale linearly with skill at extreme ends (e.g. top 100 players).
    *   *Defense*: We can replace the linear scaling $w_s \cdot (T_A - T_B)$ with a non-linear Multi-Layer Perceptron $f(T_A, T_B)$ that learns a flexible mapping of skill difference.
*   *Card Level Confounding*: Card levels are not represented in the pure matchup equation.
    *   *Defense*: In competitive tournament play, card levels are capped/equalized. Our inference API correctly assumes capped level play, fitting the product goal of evaluating matchup advantages.
