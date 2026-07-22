# Clash Royale Matchup Predictor — Comprehensive Plain-English Project Guide

## Context and Purpose

This document provides a comprehensive, jargon-free guide to the Clash Royale Matchup Predictor codebase. It is designed to help the project owner understand every file, architectural decision, and experiment well enough to defend the project in a technical interview.

All technical terms are explained in plain English when first introduced. No raw mathematical notation or formulas are used.

This guide has two parts:
1. **Part 1 — File-by-File Guide**: A step-by-step walkthrough of every non-trivial file in the repository, ordered by how a raw ladder match is collected, stored, processed, modeled, evaluated, and served.
2. **Part 2 — What We Tried That Didn't Work (or Was Rejected Before Being Tried)**: A candid, empirical account of design decisions, rejected alternatives, failed experiments, and statistical reconciliations.

---

## Part 1 — File-by-File Guide

Files are grouped logically by their role in the end-to-end pipeline.

---

### 1. Data Collection & Scraping

#### `collector/api_client.py`
- **What it is**: A network client that fetches player battle logs and ladder histories directly from Supercell's official Clash Royale API.
- **Why it exists**: Without this file, the project would have no real player battle data to train on.
- **How it works**: Sends web requests to the official API using an authentication token, handles request rate limits (pausing if too many requests are sent), and returns raw battle data in structured format.

#### `collector/collector.py`
- **What it is**: An automated crawler that discovers new players and collects thousands of ladder matches.
- **Why it exists**: Without this file, data collection would be limited to a few manually specified players.
- **How it works**: Starts with a seed list of top ladder players, inspects their recent battles to find their opponents, adds those opponents to a queue, and recursively fetches their battles.

#### `collector/parser.py`
- **What it is**: A data extractor that cleans raw API responses and pulls out only the necessary battle information.
- **Why it exists**: Without this file, raw API responses contain hundreds of unnecessary fields (like player clan badges or emote choices) that waste storage and memory.
- **How it works**: Parses nested JSON records (a standard format for transmitting web data) to extract 8-card player decks, 8-card opponent decks, trophy counts, match timestamps, and the battle result (win or loss).

#### `scrape_royale_api.py`
- **What it is**: A web scraper script that collects battle logs and card details directly from RoyaleAPI web pages.
- **Why it exists**: Serves as a backup data collector if the official API key expires or hits rate limits.
- **How it works**: Downloads public HTML web pages from RoyaleAPI, parses the card image URLs and match tables, and extracts deck combinations into local data files.

---

### 2. Database Storage & Schema

#### `database/storage.py`
- **What it is**: A database manager that saves collected players, battles, and crawler queues into a local SQLite database file.
- **Why it exists**: Without this file, collected battles would exist only in volatile memory and be lost whenever the collector script stops.
- **How it works**: Manages an SQLite database (a lightweight database stored as a single file on disk), inserting battle records while automatically ignoring duplicate matches based on unique battle timestamps and player IDs.

#### `collector/database.py`
- **What it is**: A legacy database helper module used during early collector development.
- **Why it exists**: Maintained to support older collection utility scripts without breaking legacy code.
- **How it works**: Provides basic database helper functions to query player queue status and battle counts from SQLite tables.

#### `database/check_db.py`
- **What it is**: A database diagnostic script that prints total battle counts, table structures, and data collection progress.
- **Why it exists**: Without this file, verifying database size and data health requires running manual database commands in a terminal.
- **How it works**: Connects to the local SQLite database, counts rows across tables, checks for missing values, and prints a formatted status summary.

#### `reset_database.py`
- **What it is**: A database reset utility that wipes all stored battle tables and re-initializes a clean database schema.
- **Why it exists**: Needed during testing to clear corrupt or temporary datasets and start collection from scratch.
- **How it works**: Drops existing database tables and executes creation scripts to re-establish empty tables.

---

### 3. Feature Engineering & Domain Processing

#### `features/generate_card_library.py` & `features/card_library.json`
- **What it is**: A script and resulting JSON file that maintain a master dictionary of all 122+ Clash Royale cards.
- **Why it exists**: Without this file, numeric card IDs returned by the API (e.g. 26000000) cannot be converted into card names ("Knight"), elixir costs, or model embedding indices.
- **How it works**: Reads raw card metadata from the API, sorts card IDs, assigns each card a unique integer index from zero up to the total number of cards minus one, and saves the mapping to a local JSON file.

#### `features/feature_engine.py`
- **What it is**: A domain feature calculator that computes deck summary statistics (average elixir cost, spell counts, air defense count, win condition count).
- **Why it exists**: Essential for baseline tabular machine learning models (like Logistic Regression and LightGBM) that cannot process raw card sets directly.
- **How it works**: Looks up each card in an 8-card deck against `card_library.json`, extracts numerical attributes (elixir cost, attack type), and sums them to create a numeric feature vector (a list of numbers representing deck properties).

#### `preprocessing/deck_features.py`
- **What it is**: A one-hot feature encoder that converts 8-card decks into 216-dimensional binary vectors.
- **Why it exists**: Baseline machine learning algorithms require fixed-length numerical inputs where each card's presence is represented as a 1 or 0.
- **How it works**: Creates a binary array of length 108 for each player, placing a 1 at the index of each card present in their deck, and appends trophy differences.

---

### 4. Preprocessing & DataLoaders

#### `preprocessing/preprocess.py`
- **What it is**: A data cleaning pipeline that converts raw SQLite battle records into a clean, compressed Parquet dataset file.
- **Why it exists**: Without this file, corrupted or incomplete battle records (like 2v2 matches or incomplete 7-card decks) would cause training crashes.
- **How it works**: Reads SQLite database rows, filters out non-ladder matches, verifies that both decks contain exactly 8 valid cards, converts trophy strings to numbers, and saves the cleaned dataset into Parquet format (a fast, compressed columnar file format).

#### `training/dataset.py`
- **What it is**: A PyTorch dataset module (`ClashRoyaleDataset` and `get_dataloaders`) that feeds batches of match data into the neural network during training.
- **Why it exists**: Without this file, PyTorch neural networks cannot efficiently load, batch, or shuffle match data during training, and perspective bias (favoring Player 1 over Player 2) would occur.
- **How it works**: Loads match rows, converts 8-card decks into integer tensors (PyTorch's multi-dimensional array format), and applies 50% symmetry data augmentation (randomly swapping Player 1 and Player 2 decks while negating trophy differences and flipping the win/loss target).

#### `training/splits.py`
- **What it is**: A chronological dataset splitter that divides matches into 70% training, 15% validation, and 15% testing sets based on match timestamps.
- **Why it exists**: Prevents data leakage (accidental future information influencing past predictions) by ensuring the test set strictly contains matches played after the training matches.
- **How it works**: Sorts all battles chronologically by timestamp, calculates index boundaries for 70/15/15 ratios, and slices the dataset accordingly.

#### `training/dataset_validation.py`
- **What it is**: A dataset integrity checker that verifies tensor shapes, label distributions, and index boundaries before training starts.
- **Why it exists**: Prevents silent data corruption (like out-of-bounds card indices) from causing training failures.
- **How it works**: Asserts that all card indices fall between zero and the maximum card index, checks that target labels are binary (0 or 1), and verifies that Player 1 and Player 2 deck sizes equal 8.

---

### 5. Neural Network Model Definition

#### `models/deck_encoder.py`
- **What it is**: The Deep Sets deck encoder (`DeckEncoder`) that converts an 8-card deck into a single 16-dimensional vector representation.
- **Why it exists**: Without this file, the model would treat deck card order as meaningful (e.g. treating Knight in Slot 1 differently from Knight in Slot 8).
- **How it works**: Looks up each of the 8 cards in a trainable 16-dimensional embedding table (a learned lookup matrix mapping card IDs to dense numerical vectors), passes each vector through a 2-layer Multi-Layer Perceptron (MLP, a basic neural network layer), and averages the 8 resulting vectors into a single 16-dimensional deck representation (permutation-invariant average pooling).

#### `models/interaction_head.py`
- **What it is**: The mathematical comparison module (`BradleyTerryHead` and `SkillDifference`) that calculates matchup probabilities and isolates player skill differences.
- **Why it exists**: Guarantees that self-matchups always equal 50%, guarantees that reversing deck order flips the prediction, and un-confounds player skill during training.
- **How it works**: Calculates a comparison score using a special anti-symmetric matrix. This mathematical design guarantees two physical rules: first, a deck facing itself will always predict an exact 50% tie; second, reversing the order of the two decks strictly reverses the prediction outcome. The skill difference module calculates a player trophy bias using an odd mathematical function (a function where flipping the input sign strictly flips the output sign). Setting the trophy difference to zero during evaluation completely removes the player skill effect, isolating pure deck strength.

#### `models/predictor.py`
- **What it is**: The complete Siamese neural network (`MatchupModel`) assembling `DeckEncoder`, `BradleyTerryHead`, and `SkillDifference`.
- **Why it exists**: Serves as the primary production neural network architecture.
- **How it works**: Passes Player 1 and Player 2 decks through the shared `DeckEncoder`, passes the two deck vectors through `BradleyTerryHead`, adds the `SkillDifference` bias, and applies the sigmoid function to output a win probability between 0% and 100%.

#### `models/attention_encoder.py`
- **What it is**: An alternative deck encoder (`SelfAttentionDeckEncoder`) that uses Multi-Head Self-Attention instead of average pooling.
- **Why it exists**: Created for an ablation study (an experiment testing whether removing or altering a model component hurts performance) to determine if inter-card attention improves deck encoding.
- **How it works**: Replaces simple average pooling with self-attention (a neural network mechanism allowing cards in a deck to weigh their importance relative to each other), adding 1,184 additional parameters.

---

### 6. Training & Checkpointing

#### `training/trainer.py`
- **What it is**: The training controller (`Trainer`) that executes model training, validation checks, early stopping, and checkpoint saving.
- **Why it exists**: Without this file, training logic would be duplicated across scripts, and models could overfit (memorize training data instead of learning general patterns).
- **How it works**: Loops over training batches, computes binary cross-entropy loss, updates model weights using AdamW optimizer, monitors validation loss, triggers learning rate reduction on plateaus, and saves the best model weights (`best_validation.pt`) via `CheckpointManager`.

#### `train.py`
- **What it is**: The main entry-point script executed to train `MatchupModel` on the full training dataset.
- **Why it exists**: Provides a single command to run production training end-to-end.
- **How it works**: Loads dataset configurations, instantiates `ClashRoyaleDataset`, constructs `MatchupModel`, and launches `Trainer.fit()`.

#### `training/experiment_tracker.py`
- **What it is**: An experiment logger that records hyperparameter choices, epoch losses, and test metrics to JSON log files.
- **Why it exists**: Maintains a complete audit trail of every training run.
- **How it works**: Writes structured JSON entries containing training timestamps, learning rates, loss values, and metric evaluations.

---

### 7. Evaluation & Analysis Scripts

#### `training/baselines.py`, `training/benchmarks.py` & `training/benchmark_runner.py`
- **What it is**: Baseline model modules that implement and evaluate standard machine learning algorithms (Logistic Regression, LightGBM, CatBoost, Random Guess).
- **Why it exists**: Essential for establishing baseline performance to prove whether the complex Siamese neural network actually outperforms simple standard models.
- **How it works**: Encodes 8-card decks into 216-dim one-hot vectors, applies `StandardScaler` (scaling numerical features to zero mean and unit variance) to trophy differences, and trains decision tree and linear classifiers.

#### `training/calibration.py`
- **What it is**: A probability calibration evaluator that calculates Expected Calibration Error (ECE) and Brier Score / Mean Squared Error (MSE).
- **Why it exists**: Ensures that when the model predicts a 70% win probability, the deck actually wins 70% of the time in real matches.
- **How it works**: Groups predictions into probability bins (e.g. 60%-70%), compares mean predicted probability against actual win rate in each bin, and computes weighted error.

#### `research/scripts/phase0_analysis.py`
- **What it is**: A statistical audit script that re-computes test metrics, McNemar's test, DeLong's ROC-AUC test, and Brier score confidence intervals.
- **Why it exists**: Verifies whether observed performance improvements over baseline models are statistically significant.
- **How it works**: Loads saved test set predictions, evaluates paired classification decisions, runs DeLong's test (a statistical method for comparing ROC curves), and computes 95% bootstrap confidence intervals.

#### `research/scripts/tune_lr_val.py`
- **What it is**: A hyperparameter tuning script that evaluates Logistic Regression L2 regularization values on the validation split.
- **Why it exists**: Confirms that a regularization strength of C=0.1 is validation-optimal with zero test-set leakage.
- **How it works**: Fits Logistic Regression models across regularisation values on scaled training data with 50% symmetry augmentation and evaluates validation accuracy and ROC-AUC on validation matches.

#### `research/scripts/run_learning_curve.py`
- **What it is**: A data scaling analysis script that evaluates model performance across chronological training subsamples (10k, 25k, 50k, 70k battles) across seeds 42, 43, and 44.
- **Why it exists**: Evaluates how model accuracy scales as training dataset size increases.
- **How it works**: Trains 3 random seeds per subsample size using canonical `Trainer` with validation checkpoint restoration, outputs `learning_curve.csv` metrics and `learning_curve.png` plot.

#### `research/scripts/run_attention_ablation.py`
- **What it is**: An ablation audit script comparing Deep Sets (average pooling) vs Multi-Head Self-Attention across seeds 42, 43, 44, including a paired t-test.
- **Why it exists**: Determines whether self-attention improves deck representations over Deep Sets.
- **How it works**: Trains both encoders across 3 seeds using canonical `Trainer`, computes per-seed paired differences, and evaluates paired t-statistic (t = 0.3956, p = 0.7306).

#### `research/scripts/evaluate_overlap_buckets.py`
- **What it is**: An analysis script that evaluates model predictions across exact card overlap buckets (0 to 8 shared cards).
- **Why it exists**: Investigates model behavior on mirror/high-overlap matchups.
- **How it works**: Filters test matches by exact shared card count, computes accuracy and mean predicted probability per bucket, and verifies centering around 0.50.

#### `research/scripts/inspect_embeddings.py`
- **What it is**: A card representation inspector that extracts 16-dimensional card embeddings and computes cosine similarity matrices.
- **Why it exists**: Checks whether the model learned meaningful card concepts (e.g. assigning high vector similarity to similar win conditions like Giant and Golem).
- **How it works**: Passes card indices through trained `DeckEncoder` weights, calculates cosine similarities (measuring vector angle closeness between 1.0 and -1.0), and prints top card pairs.

#### `research/scripts/export_onnx.py`
- **What it is**: An ONNX export utility that converts the trained PyTorch model into ONNX format (`models/onnx/matchup_model.onnx`).
- **Why it exists**: Allows serving predictions in production environments using lightweight ONNX Runtime without needing PyTorch installed.
- **How it works**: Passes dummy input tensors through `MatchupModel`, traces computational graph, and exports binary ONNX model file.

#### `research/scripts/generate_pdf.py`
- **What it is**: A document compilation script that compiles `docs/paper.md` into a publication PDF.
- **Why it exists**: Automatically builds formatted PDF research papers.
- **How it works**: Parses markdown structure, formats tables, headers, and text using FPDF2 library, and writes `docs/paper.pdf`.

---

### 8. Serving API & Web Server

#### `api/app.py`
- **What it is**: A production FastAPI web server exposing REST endpoints (`/predict`, `/health`, `/card-embedding`, `/explain`).
- **Why it exists**: Allows external applications, websites, and user interfaces to request real-time matchup predictions.
- **How it works**: Loads trained model weights at server startup, parses incoming JSON deck lists, sets trophy difference to zero to isolate intrinsic matchup probability, and returns predictions with execution timing.

---

### 9. Test Suite

#### `tests/test_api_sprint18.py`
- **What it is**: Unit tests for the FastAPI web server.
- **Why it exists**: Ensures REST endpoints respond correctly and handle invalid inputs without crashing.
- **How it works**: Uses FastAPI test client to send mock HTTP POST requests to `/predict` and checks response status codes.

#### `tests/test_baselines_sprint11.py`
- **What it is**: Unit tests for Logistic Regression, LightGBM, and Random Guess baseline runners.
- **Why it exists**: Verifies baseline model training pipelines and data format transformations.
- **How it works**: Fits baseline classifiers on synthetic data batches and asserts output metric keys exist.

#### `tests/test_card_library.py`
- **What it is**: Unit tests for `card_library.json` integrity.
- **Why it exists**: Prevents missing card IDs or corrupt mappings.
- **How it works**: Checks that card library contains expected cards, unique contiguous indices, and valid elixir costs.

#### `tests/test_database.py`
- **What it is**: Unit tests for SQLite database operations.
- **Why it exists**: Verifies database connection, table creation, and transaction safety.
- **How it works**: Creates temporary SQLite test database, inserts mock battles, and asserts row counts match.

#### `tests/test_dataset_sprint10.py`
- **What it is**: Unit tests for `ClashRoyaleDataset` and 50% symmetry data augmentation.
- **Why it exists**: Confirms that deck swapping correctly flips targets and negates trophy diffs.
- **How it works**: Passes known deck pairs through dataset with augmentation probability of 1.0 and asserts deck swapping, trophy negation, and target inversion.

#### `tests/test_deck_encoder_sprint12.py`
- **What it is**: Unit tests for `DeckEncoder` permutation invariance.
- **Why it exists**: Guarantees deck encoding is order-independent.
- **How it works**: Passes a deck and a shuffled version of the same deck through `DeckEncoder` and asserts output vectors are identical.

#### `tests/test_matchup_model_sprint13.py`
- **What it is**: Unit tests for Siamese `MatchupModel` mathematical properties.
- **Why it exists**: Verifies self-matchup tie guarantees and anti-symmetry.
- **How it works**: Passes identical decks through model and asserts prediction equals exactly 0.5000 (50% probability).

#### `tests/test_preprocessing.py`
- **What it is**: Unit tests for raw JSON parsing and data cleaning.
- **Why it exists**: Ensures invalid raw battle records are correctly filtered.
- **How it works**: Passes valid and invalid raw battle dictionaries to parser and verifies clean output.

#### `tests/test_trainer_sprint14.py`
- **What it is**: Unit tests for `Trainer`, early stopping, and checkpoint restoration.
- **Why it exists**: Verifies model training loop execution and checkpoint saving.
- **How it works**: Runs `Trainer` for 2 epochs on dummy data and verifies checkpoint files are written to disk.

---

### 10. Documentation

- `README.md`: Main repository documentation detailing setup, metrics, and architecture.
- `MODEL_CARD.md`: Production model specifications, intended use, ethical considerations, and limitations.
- `docs/paper.md`: Academic research paper detailing Siamese architecture, mathematical proofs, and results.
- `docs/paper.pdf`: Compiled PDF publication document.
- `research/notes/FINAL_PROJECT_REPORT.md`: Comprehensive end-to-end engineering report.
- `research/notes/DECISION_HISTORY.md`: High-yield 30-second summary list of all project decisions for interview preparation.

---

## Part 2 — What We Tried That Didn't Work (or Was Rejected Before Being Tried)

This section details every design alternative, failed experiment, rejected proposal, and statistical reconciliation in plain English, backed by empirical data from the repository.

---

### 1. A Broken Early Pipeline That Looked Like It Was Working
- **What Happened**: Early in development, the preprocessing pipeline passed all unit tests cleanly. However, the test dataset contained only 4 battles. This sample was too small to include certain cards (such as "P.E.K.K.A." versus "Mini P.E.K.K.A.") whose string names differed between data files. As soon as the pipeline ran on a real dataset containing all cards, key lookups crashed immediately.
- **Why Rejected / Fixed**: Fixed by expanding test coverage (`tests/test_card_library.py`) to test every single card in `card_library.json`.
- **Key Takeaway**: Passing unit tests on a tiny sample does not mean code works; it often means the sample never exercised the bug.

---

### 2. A Hand-Built Interaction Knowledge Base of Card Counters
- **What Was Considered**: Building a manual lookup matrix specifying which cards counter which other cards across all ~122 Clash Royale cards.
- **Why Rejected / Current Status**: This idea was discussed during early architectural design and redirected before any implementation was built. No hand-built counter matrix file was ever created. It was rejected because it would require thousands of subjective entries that would become outdated every balance patch, and because counter relationships are automatically learned from card attributes and vector embeddings.

---

### 3. Representing Decks as Graphs using Graph Neural Networks (GNNs)
- **What Was Considered**: Modeling an 8-card deck as a graph (a mathematical network of connected nodes) and processing it with a Graph Neural Network (GNN).
- **Why Rejected**: A Clash Royale deck is an unordered collection of 8 cards with no fixed connections or structural edges (unlike social networks, transportation grids, or chemical molecules). There are no real graph edges to exploit. Deep Sets (permutation-invariant average pooling over card embeddings) is mathematically tailored for unordered set inputs, computationally lighter, and easier to optimize.

---

### 4. A Simple Deck Rating Number (like Chess Elo)
- **What Was Considered**: Assigning a single scalar numerical rating (like an Elo score) to each deck to predict matchup winners.
- **Why Rejected**: Clash Royale matchups exhibit non-transitive rock-paper-scissors dynamics (Deck A beats Deck B, Deck B beats Deck C, Deck C beats Deck A). A single numerical rating forces all decks onto a single 1-dimensional line from "weak" to "strong," which mathematically cannot represent cyclic matchup advantages.

---

### 5. Feeding Deck Summaries Side-by-Side (Concatenation) without a Special Comparison Step
- **What Was Considered**: Feeding concatenated deck vectors directly into a standard Multi-Layer Perceptron (MLP) classification network.
- **Why Rejected**: Unstructured MLPs only hope to learn symmetry properties from data, risking perspective bias (where Player 1 winning vs Player 2 predicts differently than Player 2 losing to Player 1) and self-matchup errors. The production `BradleyTerryHead` uses an anti-symmetric comparison matrix that mathematically guarantees that self-matchups always predict an exact 50% tie and reversing deck order strictly flips prediction probability by construction.

---

### 6. Removing Player Trophy/Skill Information from Training Entirely
- **What Was Considered**: Stripping player trophy counts from training data to isolate pure deck strength.
- **Why Rejected**: Confounding error. Omitting player skill does not remove its influence; it turns skill into an unobserved confounding variable that biases learned deck representations. The correct approach (`SkillDifference`) includes trophy difference during training and sets trophy difference to zero at evaluation time, isolating intrinsic matchup probability.

---

### 7. Multi-Head Self-Attention Encoder vs Deep Sets Average Pooling
- **What Was Tested**: Replacing Deep Sets average pooling with a Multi-Head Self-Attention encoder (`SelfAttentionDeckEncoder`).
- **Initial Observation**: A single training run produced 59.16% accuracy (+1.04 percentage points over baseline).
- **Multi-Seed Audit Verification**: Re-evaluating across seeds 42, 43, and 44 using canonical `Trainer` with validation checkpoint restoration showed:
  - **Deep Sets (Average Pooling)**: 58.78% ± 0.20% accuracy
  - **Self-Attention Encoder**: 58.93% ± 0.45% accuracy (mean difference +0.15 percentage points)
  - **Paired t-Test across seeds**: Per-seed differences of -0.27, -0.21, and +0.92 percentage points yielded a paired t-statistic of t = 0.3956 and p-value of p = 0.7306 (statistically non-significant).
- **Decision**: **REJECTED**. The +0.15 percentage point mean accuracy gain is statistically non-significant and well within seed noise. Adding 1,184 parameters and self-attention computational overhead for zero dependable benefit is not justified.

---

### 8. Early Misconception Regarding Mirror Matchup Errors
- **What Happened**: Early analysis speculated that poor performance on mirror matchups (decks sharing 6 or more cards) was due to the model lacking card play-order information.
- **Why Rejected**: The model never had access to play-order data in any match. Evaluating test matches by exact shared card count (`research/scripts/evaluate_overlap_buckets.py`) revealed:
  - For high-overlap decks (6 or more shared cards, 40 total matches), accuracy is 55.00% (22/40 correct).
  - The mean predicted probability is 0.5142 against a mean actual win rate of 0.5250.
  - Probabilities are appropriately centered around 0.50, ruling out ordering or sign bugs. High accuracy variance in high-overlap buckets is caused by small sample sizes (19 or fewer matches per individual overlap level), not structural flaws.

---

### 9. Unaugmented / Unscaled Baseline Re-Check Reconciliation
- **What Happened**: An early baseline re-check evaluated Logistic Regression without data augmentation or feature scaling, yielding an inferior 56.80% accuracy and 0.6125 ROC-AUC.
- **Reconciliation**: Re-evaluating with 50% symmetry data augmentation (`augment_prob = 0.5`) and `StandardScaler` on trophy difference restored true baseline metrics: **58.08% accuracy, 0.6169 ROC-AUC, and 0.2388 Brier score**.
- **Side-by-Side Significance Findings**:
  1.  **Accuracy Edge (+0.04 percentage points)**: **Not statistically significant** (McNemar p = 0.3389, 95% confidence interval -0.28% to +0.78%).
  2.  **ROC-AUC (+0.0093)**: **Statistically significant** (DeLong test p < 0.001, 95% confidence interval +0.0073 to +0.0141).
  3.  **Brier Score (-0.0015)**: **Statistically significant** (Reconciled Mean Squared Error confidence interval 0.0013 to 0.0025).

---

### 10. Simulating Balance Patches by Editing Card Stats
- **What Was Considered**: Simulating hypothetical balance patches (e.g. "what if Card X gets a 10% damage buff") by editing card input vectors.
- **Why Rejected**: `DeckEncoder` represents each card as a discrete categorical embedding index, not a vector of physical stats (HP, DPS, speed). Stat-edit balance simulation is impossible without an entirely different architecture.

---

### 11. Four-Part Microservice Backend Architecture
- **What Was Considered**: Splitting backend infrastructure into 4 separate microservices (prediction, explanation, recommendation, embedding lookup).
- **Why Rejected**: Over-engineering for a single model application. A monolithic FastAPI server (`api/app.py`) serves all endpoints with far lower deployment overhead, zero network latency between components, and simpler maintenance.

---

### 12. False Authorship Attribution Correction
- **What Happened**: Early generated report drafts mistakenly included "DeepMind Advanced Agentic Coding Team" in header credits and PDF citations.
- **Correction**: Identified during code audit and updated across all repository markdown, paper, license, and script files to `[AUTHOR NAME]` to maintain strict authorship integrity.

---

### 13. Seed Reproducibility Discrepancy Investigation Status
- **Investigation Status**: **RESOLVED**.
- **Root Cause**: `run_attention_ablation.py` used the `Trainer` class with `CheckpointManager` and `EarlyStopping` (restoring the best validation loss checkpoint `best_validation.pt` at training end) and passed a seeded generator to `get_dataloaders`. In contrast, `run_learning_curve.py` ran an unmonitored raw 10-epoch loop without validation monitoring or checkpoint restoration, and used unseeded batch shuffling.
- **Resolution**: `run_learning_curve.py` was updated to use the canonical `Trainer` + `get_dataloaders` pipeline with validation checkpoint restoration. Running both scripts now yields **100% identical Deep Sets results across both scripts**:
  - Seed 42: **59.06%** Accuracy / **0.6284** ROC-AUC
  - Seed 43: **58.68%** Accuracy / **0.6300** ROC-AUC
  - Seed 44: **58.62%** Accuracy / **0.6311** ROC-AUC
- **Current Paired t-Test**: The paired t-test result of t = 0.3956 and p = 0.7306 is the final, fully reproducible statistic calculated directly from these synchronized per-seed outputs.

---

## Final Verification Checklist

- [x] G1: "14,884" number removed from hand-built counter matrix entry; entry states plainly that no implementation was ever built.
- [x] G2: All raw equations, LaTeX formulas, and unexplained symbols removed and replaced with plain-English sentences.
- [x] G3: E1 seed-reproducibility investigation status documented as RESOLVED with root cause and synchronized metrics (Seed 42: 59.06%, Seed 43: 58.68%, Seed 44: 58.62%, t = 0.3956, p = 0.7306).
- [x] G4: `research/notes/DECISION_HISTORY.md` confirmed present and synchronized.
