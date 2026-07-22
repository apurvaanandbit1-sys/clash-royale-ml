# Phase 2 Design Specification & System Boundaries

This document outlines the design specifications for feature explanation, counter-suggestions, deck optimization, and meta exploration, alongside explicit scope boundaries.

---

## 1. Feature Attribution & Explanations

### 1.1 Methodology: Integrated Gradients
Rather than using SHAP (which scales poorly for continuous set embeddings and creates out-of-distribution baseline samples), we select **Integrated Gradients** (Sundararajan et al., 2017).

Given a deck $D_A = \{c_1, \dots, c_8\}$, Integrated Gradients computes the path integral of gradients along the straight-line trajectory from a baseline card set $D_0$ to $D_A$:
$$\text{Attribution}_i = (E(c_i) - E(c_0)) \times \int_{0}^{1} \frac{\partial \text{MatchupModel}(\alpha E(c_i) + (1-\alpha) E(c_0))}{\partial E(c_i)} d\alpha$$

### 1.2 Output Representation
Attributions are grouped into card-level influence scores representing how much each individual card contributes to the win probability (positive = favorable counter, negative = vulnerable card).

---

## 2. Counter-Suggestions & Confidence Intervals

### 2.1 Single-Card Swap Engine
Given a user's deck $D_A$, the system evaluates single-card substitutions $c_i \rightarrow c_k'$ across candidate cards $c_k' \notin D_A$.

### 2.2 Confidence Bands
Because the model has a modest ROC-AUC ($\approx 0.6262$), point estimates alone ($p=0.54$) can be misleading. Every counter suggestion is accompanied by a **95% Bootstrap Confidence Interval**:
$$\hat{p} \pm 1.96 \times \text{SE}_{\text{bootstrap}}$$
Suggestions are categorized as:
*   **High Confidence Improvement**: Lower bound of suggestion CI $> \text{Upper bound of current deck CI}$.
*   **Marginal / Uncertain Improvement**: Overlapping CIs.

---

## 3. Multi-Card Deck Optimizer

### 3.1 Overfitting Risk Control
Searching all 200,000+ candidate two-card or three-card swaps creates a high multiple-testing risk. Point-estimate rankings will overfit to model noise.

### 3.2 CI Exclusion Filter
The Deck Optimizer enforces a strict significance filter:
1.  Compute baseline win probability CI $[L_0, U_0]$ for current deck $D_A$.
2.  For each candidate swap $D_{A'}$, estimate CI $[L', U']$.
3.  **Filter Rule**: Surface $D_{A'}$ if and only if $L' > U_0$ (the lower bound of the candidate CI strictly excludes the upper bound of the current deck estimate).

---

## 4. Explicitly Out of Scope (System Boundaries)

The following capabilities are explicitly out of scope for this architecture:

1.  **Balance Patch Simulator**: The ID-indexed card embedding table $E(c_i)$ maps discrete card IDs to latent vectors. It has no mechanism to query counterfactual stat modifications (e.g. "+5% HP to Knight"). Balance patch simulation requires stat-based parametric features, not ID embeddings.
2.  **Dynamic Match Graph Neural Networks (GNNs)**: Modeling spatial troop interactions on dynamic grids requires in-battle placement and replay logs. The repository scope is strictly bounded to pre-match 8-card composition.
3.  **Graph Database Stores (e.g. Neo4j)**: At a scale of 122 cards and 8-card sets, in-memory Python dictionaries and SQLite are orders of magnitude faster.
4.  **Microservice Architecture Splitting**: The application is deployed as a single unified FastAPI backend (`api/app.py`). Splitting into microservices adds operational latency without scaling benefits at this throughput level.
