# Clash Royale Matchup Predictor — Decision & Experiment History

This document lists every major engineering decision, implemented feature, rejected proposal, and failed experiment in the project. Format: concise bullet points (2–4 lines maximum per entry) for interview preparation.

---

## Section 1 — What Was Implemented and Shipped

- **Deep Sets Deck Encoder (`models/deck_encoder.py`)**
  - **Why**: Standard neural networks treat input order as meaningful, but Clash Royale decks are unordered 8-card sets.
  - **How**: Maps card IDs to 16-dim embeddings, processes each through a 2-layer MLP, and averages vectors across the 8 cards (permutation-invariant average pooling).
  - **Status**: Production

- **Skew-Symmetric Matchup Head (`models/interaction_head.py`)**
  - **Why**: Unstructured classifiers risk predicting self-matchups not equal to 50% or perspective-dependent probabilities.
  - **How**: Uses an anti-symmetric comparison matrix that mathematically guarantees self-matchups predict an exact 50% tie and reversing deck order strictly reverses win probability.
  - **Status**: Production

- **Skill-Difference Confounder Head (`models/interaction_head.py`)**
  - **Why**: Player trophy differences confound intrinsic deck strength during training on public ladder matches.
  - **How**: Models skill bias using an odd mathematical function during training, and sets trophy difference to zero at prediction time to isolate intrinsic deck strength.
  - **Status**: Production

- **Chronological Train / Validation / Test Split (`training/splits.py`)**
  - **Why**: Random splitting causes temporal data leakage by evaluating models on past matches using future training data.
  - **How**: Sorts 100,542 matches chronologically by timestamp into 70% train (70,379), 15% validation (15,081), and 15% test (15,082) splits.
  - **Status**: Production

- **50% Symmetry Data Augmentation (`training/dataset.py`)**
  - **Why**: Public match logs list arbitrary player perspectives, causing neural models to develop Player 1 / Player 2 position bias.
  - **How**: Randomly swaps Player 1 and Player 2 decks with 50% probability (swapping decks, negating trophy diffs, and flipping win targets) during batch creation.
  - **Status**: Production

- **Baseline Model Suite (`training/baselines.py`, `training/benchmark_runner.py`)**
  - **Why**: Necessary to quantify exact performance gains added by complex Siamese neural architectures over traditional ML.
  - **How**: Encodes decks as 216-dim one-hot vectors with scaled trophy diffs, training Logistic Regression (58.08% Acc / 0.6169 AUC / 0.2388 Brier), LightGBM (57.97% Acc), CatBoost (57.62% Acc), and Random Guess (50.36% Acc).
  - **Status**: Production (Benchmarking)

- **Statistical Rigor & Significance Audit (`research/scripts/phase0_analysis.py`)**
  - **Why**: Claiming model superiority based on raw test accuracy without hypothesis testing is scientifically invalid.
  - **How**: Runs McNemar's test (p = 0.3389, non-significant accuracy edge), DeLong's ROC-AUC test (p < 0.001, significant), and 95% bootstrap CIs on Brier MSE diff (0.0015, significant).
  - **Status**: Production (Audit Completed)

- **Multi-Seed Validation Protocol (`research/scripts/run_learning_curve.py`, `research/scripts/run_attention_ablation.py`)**
  - **Why**: Single training runs are subject to random seed variance, leading to false architectural conclusions.
  - **How**: Trains all comparative architectural variants across seeds 42, 43, and 44 using canonical `Trainer` and reports mean and standard deviation.
  - **Status**: Production Standard

- **ONNX Model Export (`research/scripts/export_onnx.py`, `models/onnx/matchup_model.onnx`)**
  - **Why**: Serving predictions in production using full PyTorch runtimes introduces unnecessary memory and latency overhead.
  - **How**: Traces model computational graph with dummy input tensors and exports binary ONNX runtime weights.
  - **Status**: Production

- **FastAPI Production Web Server (`api/app.py`)**
  - **Why**: External clients and user interfaces require real-time HTTP endpoints for matchup predictions and explanations.
  - **How**: Exposes REST endpoints (`/predict`, `/health`, `/card-embedding`, `/explain`) with in-memory model loading, categorical confidence bands, and meta-deck caching.
  - **Status**: Production

- **Automated Card Library Generator (`features/generate_card_library.py`, `features/card_library.json`)**
  - **Why**: Card IDs returned by official APIs require canonical mapping to dense embedding indices.
  - **How**: Fetches raw card metadata, sorts card IDs, creates contiguous integer indices (0 up to max card index), and exports master JSON dictionary.
  - **Status**: Production

---

## Section 2 — What Was Tried and Didn't Work, or Was Rejected Before Being Tried

- **Broken Early 4-Battle Test Pipeline**
  - **Why considered**: Rapid initial verification of raw battle log parsing and preprocessing.
  - **What happened / why rejected**: A 4-battle test sample passed unit tests but missed card name mismatches ("P.E.K.K.A." vs "Mini P.E.K.K.A.") across data files, causing real pipeline crashes. Fixed by expanding unit tests to validate all 122+ cards.
  - **Status**: Corrected

- **Hand-Built Card Counter Knowledge Base**
  - **Why considered**: Manual matrix of card counter relationships across all ~122 Clash Royale cards.
  - **What happened / why rejected**: Considered during early design and redirected before any implementation was built (no file was created). Rejected due to balance patch churn and redundancy with learned card embeddings.
  - **Status**: Rejected (No Implementation Built)

- **Graph Neural Network (GNN) Deck Representation**
  - **Why considered**: Modeling an 8-card deck as a graph with inter-card node connections.
  - **What happened / why rejected**: Rejected because an 8-card deck is an unordered set with no fixed graph topology or physical edges; Deep Sets (permutation-invariant set pooling) is mathematically superior.
  - **Status**: Rejected

- **Single Scalar Deck Strength Rating (Elo-Style)**
  - **Why considered**: Assigning a 1D numerical strength score to each deck.
  - **What happened / why rejected**: Rejected because Clash Royale exhibits non-transitive rock-paper-scissors matchup cycles (Deck A > Deck B > Deck C > Deck A), which cannot be represented on a 1D linear rating scale.
  - **Status**: Rejected

- **Side-by-Side Deck Embedding Concatenation**
  - **Why considered**: Feeding concatenated deck vectors directly into a standard MLP classifier.
  - **What happened / why rejected**: Rejected because unstructured MLPs only hope to learn symmetry from data; replaced by anti-symmetric matchup head which mathematically guarantees 50% self-ties and anti-symmetric probabilities.
  - **Status**: Rejected

- **Removing Player Trophy / Skill Data Entirely from Training**
  - **Why considered**: Stripping player trophies from training data to isolate pure deck strength.
  - **What happened / why rejected**: Rejected because omitting skill creates unobserved confounding; the correct solution (`SkillDifference`) models trophy diff during training and zeroes it at prediction.
  - **Status**: Rejected

- **Multi-Head Self-Attention Deck Encoder (`models/attention_encoder.py`)**
  - **Why considered**: Testing whether self-attention between cards improves deck encoding over average pooling.
  - **What happened / why rejected**: Single-run showed +1.04%p gain (59.16%), but 3-seed benchmark (Deep Sets: 58.78% ± 0.20% vs Self-Attention: 58.93% ± 0.45%, paired t-test t = 0.3956, p = 0.7306) proved it was seed noise. Rejected due to 1,184 extra parameters for zero dependable gain.
  - **Status**: Tested and Not Adopted

- **Play-Order Ignorance Explanation for Mirror Matchups**
  - **Why considered**: Speculated that weak accuracy on high-overlap decks (6 or more shared cards) stemmed from ignoring card play order.
  - **What happened / why rejected**: Corrected after overlap bucket evaluation showed predictions for high-overlap decks (n=40) are properly centered at 0.5142 (ruling out ordering bugs); high accuracy variance was driven by small sample sizes (19 or fewer matches).
  - **Status**: Corrected

- **Unaugmented / Unscaled Baseline Re-Evaluation Mismatch**
  - **Why considered**: Early baseline re-check evaluated Logistic Regression without data augmentation or feature scaling, yielding 56.80% Acc / 0.6125 AUC.
  - **What happened / why rejected**: Traced to missing settings; restoring 50% symmetry augmentation and scaler yielded true baseline: 58.08% Acc / 0.6169 AUC / 0.2388 Brier. McNemar test (p=0.3389) confirmed accuracy edge is non-significant, while ROC-AUC (p<0.001) is significant.
  - **Status**: Corrected

- **Simulating Balance Patches by Editing Input Stat Vectors**
  - **Why considered**: Simulating hypothetical balance patch stat buffs (e.g. +10% damage) by modifying input vectors.
  - **What happened / why rejected**: Identified as not feasible without total architecture redesign, because `DeckEncoder` represents cards by discrete categorical ID embeddings, not continuous physical stat vectors.
  - **Status**: Rejected

- **Four-Service Microservice Backend Architecture**
  - **Why considered**: Splitting backend into 4 microservices (prediction, explanation, recommendation, embedding lookup).
  - **What happened / why rejected**: Rejected as unnecessary complexity for a single model; monolithic FastAPI server (`api/app.py`) serves all endpoints with lower deployment overhead and zero inter-service network latency.
  - **Status**: Rejected

- **False Authorship Attribution in Report Headers**
  - **Why considered**: Automated report generation templates included "DeepMind Advanced Agentic Coding Team" in header credits.
  - **What happened / why rejected**: Caught during code audit and updated across all repository markdown, paper, license, and script files to `[AUTHOR NAME]` to maintain strict authorship integrity.
  - **Status**: Corrected

- **Seed-Reproducibility Discrepancy Between Analysis Scripts**
  - **Why considered**: `run_attention_ablation.py` and `run_learning_curve.py` reported slightly different Deep Sets metrics for identical seeds.
  - **What happened / why rejected**: Resolved. `run_learning_curve.py` was updated to use canonical `Trainer` + `get_dataloaders`, achieving 100% identical Deep Sets results across both scripts (Seed 42: 59.06%, Seed 43: 58.68%, Seed 44: 58.62%, paired t-test t = 0.3956, p = 0.7306).
  - **Status**: Resolved

---

## Section 3 — Still Open, Not Yet Done

- **Phase 2 Backend Features (Counter Suggestions, Deck Optimizer, Meta Explorer)** — Designed conceptually, not built.
- **Web Frontend Interface** — Not yet built.
- **Repository Directory Restructuring** — Deferred.
