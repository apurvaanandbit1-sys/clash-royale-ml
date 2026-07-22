# Clash Royale ML Project Handover Report

This document serves as the authoritative project handover and audit report for the Clash Royale Machine Learning repository. It outlines the technology stack, current architectural design, implementation details of each component, feature engineering processes, card knowledge database, known bugs, technical debt, dead code analysis, repository health scores, and recommended next steps for development.

---

## 1. Executive Summary

- **Project Identification**: Clash Royale Battle Prediction ML Pipeline.
- **Objective**: Build a robust, end-to-end predictive pipeline that crawls player match histories from the official Clash Royale API, stores matches in an SQLite database, processes the matches (augmenting them with domain-specific features and archetypes), and trains an XGBoost classifier to predict battle outcomes based on deck composition.
- **Machine Learning Problem**: A supervised binary classification task. Given two 8-card decks (Player 1 vs. Player 2), predict whether Player 1 will win the match (`win = 1`) or lose (`win = 0`). The model must learn complex non-linear interactions, deck synergies, card counters, macro-archetype matchups, and elixir efficiency curves.
- **Current Development Stage**: Late Prototype / Broken Pre-Production.
  - **Data Collection (Collector & Database)**: Stable and operational. Capable of executing a breadth-first search (BFS) crawl loop starting from seed players, parsing match logs, and indexing records.
  - **Feature Engineering**: Feature extraction is defined but currently blocked from execution due to naming mismatches on certain cards.
  - **Preprocessing**: Non-functional. The pipeline is blocked by a missing dependency module (`archetype_features.py`) and database pathing mismatches.
  - **Model Training**: A training script (`train_xgboost.py`) is written but cannot run due to the lack of preprocessed training data. It also contains hardcoded parameters (such as class imbalance weights) that need to be computed dynamically.

---

## 2. Repository Structure

Below is a detailed analysis of the repository's directories:

### `collector/`
- **Purpose**: Expose clients for the Clash Royale API, parse raw API battle logs into structured dictionaries, and orchestrate the breadth-first crawler queue.
- **Important Files**:
  - `api_client.py`: Handles HTTP requests, authentication, and rate limiting.
  - `collector.py`: Orchestrates the BFS crawling loop.
  - `database.py`: Collector-specific SQLite interface wrapper.
  - `parser.py`: Decodes raw battle log payloads.
  - `test_api.py`: A basic diagnostic script for testing player data retrieval.
  - `queue_manager.py`: Empty placeholder (0 bytes).
- **Completion Status**: **90%** (Operational, but `queue_manager.py` is empty, and the parser has index assumptions that limit it to 1v1 modes).

### `database/`
- **Purpose**: Handle SQLite schema initialization, table creation, indexing, and debugging.
- **Important Files**:
  - `storage.py`: Creates the tables (`battles`, `players`, `crawl_queue`) and sets up indexes.
  - `check_db.py`: Quick diagnostic utility to check the schema of `clashroyale.db`.
- **Completion Status**: **95%** (Stable database schema, but diagnostic helper scripts use hardcoded local paths rather than resolved paths relative to the project root).

### `features/`
- **Purpose**: Curate structural, combat, and strategic features for individual cards, and expose the main feature extraction engine.
- **Important Files**:
  - `feature_engine.py`: Computes average elixir, durability, damage index, and card counts.
  - `card_library.json`: Master registry of all cards pulled from the API (id, name, rarity, elixir, URLs).
  - `knowledge/card_knowledge.json`: Expert-coded strategic and combat tags for all 121 cards.
  - `card_categories/structural.py`: Exposes helper functions for card attributes.
  - `card_categories/{archetypes, combat, strategic, synergies}.py`: Empty placeholders (0 bytes).
  - `deck_similarity.py` & `generate_metadata.py`: Empty placeholders (0 bytes).
- **Completion Status**: **70%** (Knowledge base is complete, but category files are empty stubs, and name mismatches on cards like P.E.K.K.A trigger fatal KeyErrors).

### `preprocessing/`
- **Purpose**: Clean and normalize database records, extract deck-level features, classify decks by archetype, and output training matrices.
- **Important Files**:
  - `deck_features.py`: Maps `FeatureEngine` outputs across player and opponent decks in a pandas DataFrame.
  - `preprocess.py`: Intended to clean, merge meta-archetypes, and output Parquet matrices.
- **Completion Status**: **30%** (Broken pipeline. `preprocess.py` attempts to import a missing module `archetype_features`, and hardcodes incorrect database paths).

### `training/`
- **Purpose**: Store model training, hyperparameter tuning, and cross-validation scripts.
- **Completion Status**: **0%** (Directory exists but is empty. The actual training script `train_xgboost.py` is misplaced in the repository root).

### `models/`
- **Purpose**: Store serialized XGBoost models (e.g., JSON or pickle formats) for inference.
- **Completion Status**: **0%** (Directory exists but is empty).

### `utils/`
- **Purpose**: Helper functions, logging setups, and data format converts.
- **Completion Status**: **0%** (Directory exists but is empty).

### `docs/`
- **Purpose**: Handover documents and research materials.
- **Important Files**:
  - `research/card_knowledge_guidelines.md`: Outlines rules for populating the card knowledge base.
  - `research/gemini_cards_batch1.md`: Methodological framework and analysis for card tags.
  - `research/antigravity_task1.md` / `antigravity_walkthrough.md`: Developer tracking files.
- **Completion Status**: **80%** (Informative, but missing a formalized handover report, which this file resolves).

---

## 3. Current Pipeline

Below is an ASCII flow diagram showing the current pipeline data flow, highlighting implemented components, missing blocks, and the critical failure points:

```text
       [ Clash Royale API ]
                │
                ▼ (requests / api_client.py)
        [ Battle Parser ] (parser.py)  <-- (Assumption: 1v1 mode only)
                │
                ▼ (sqlite3 / database.py)
   [ SQLite DB: data/clashroyale.db ] <── [ check_db.py ] / [ reset_database.py ]
                │
                │   (preprocess.py)
                ├── [ CRITICAL BUG: Hardcoded db path looks in root, not data/ ]
                │
                ▼
      [ Preprocessing Pipeline ]  <─── [ archetype_features.py ] (MISSING BLOCK)
                │
                ▼
 [ Parquet: matches_with_archetypes.parquet ] (BLOCKED - Cannot be generated)
                │
                ▼ (pandas / train_xgboost.py)
       [ Feature Matrix X ]       <─── [ FeatureEngine ] (extracts 12 features)
                │                             │
                │                             └── [ CRITICAL BUG: KeyError on PEKKA ]
                ▼
     [ XGBoost Classifier ]       <─── [ scale_pos_weight ] (Hardcoded 303/403 ratio)
                │
                ▼ (Stratified 5-Fold CV)
     [ Held-out Predictions ]
```

---

## 4. Implemented Components

A complete audit of every active module in the repository:

### `collector/api_client.py`
- **Purpose**: Accesses the official Clash Royale API, handles Bearer authentication, and incorporates request throttling and error retries.
- **Public Functions/Classes**:
  - `ClashRoyaleAPI`: Main client wrapper.
    - `_request(endpoint)`: Executed requests with exponential-backoff retries for status codes `429` (rate limits) and `500+` (server errors).
    - `get_player(player_tag)`: Returns profile details.
    - `get_battle_log(player_tag)`: Fetches match logs.
    - `search_clans(name, limit)`: Searches clans.
    - `get_clan_members(clan_tag)`: Lists members in a clan.
    - `get_cards()`: Fetches global card stats.
- **Completeness**: Complete.
- **Dependencies**: `requests`, `dotenv`, `os`, `time`.

### `collector/collector.py`
- **Purpose**: Runs the BFS crawl loop. Enqueues seed players, retrieves logs, parses matches, extracts opponents, and queues those with >=8,500 trophies up to depth limits.
- **Public Functions/Classes**:
  - `load_seed_players()`: Reads from `seed_players.txt`.
  - `initialize_queue(db)`: Seeds the database queue if empty.
  - `main()`: Runs the BFS crawler loop.
- **Completeness**: Complete.
- **Dependencies**: `config.settings`, `pathlib`, `time`, `api_client`, `database`, `parser`.

### `collector/database.py`
- **Purpose**: Low-level SQLite database context manager supporting transactions, inserts, updates, and queue checks.
- **Public Functions/Classes**:
  - `Database`: Connection manager wrapper. Exposes methods: `add_player`, `player_exists`, `mark_processed`, `enqueue_player`, `get_next_player`, `set_processing`, `finish_player`, `battle_exists`, `add_battle`, `add_battles`, `get_stats`, `clear_players`, `clear_battles`, `clear_queue`, `queue_empty`.
- **Completeness**: Complete.
- **Dependencies**: `sqlite3`, `pathlib`.

### `collector/parser.py`
- **Purpose**: Standardizes raw battle JSON logs into dictionary records.
- **Public Functions/Classes**:
  - `BattleParser`: Exposes static methods:
    - `_extract_deck(cards)`: Extracts and sorts card IDs, returning a serialized JSON string.
    - `_generate_battle_id(...)`: Generates a deterministic SHA-256 hash using sorted tags and match timestamp.
    - `parse(battle)`: Parses a match record, extracting trophies, crowns, decks, and victory labels.
- **Completeness**: Partial (assumes `team[0]` and `opponent[0]` are always populated; fails/limits 2v2 modes).
- **Dependencies**: `json`, `hashlib`.

### `database/storage.py`
- **Purpose**: Standard database creation script defining tables and index trees.
- **Public Functions/Classes**:
  - `create_database()`: Creates tables `battles`, `players`, `crawl_queue` and multiple performance indexes.
- **Completeness**: Complete.
- **Dependencies**: `sqlite3`, `pathlib`.

### `database/check_db.py`
- **Purpose**: Inspects the schema of `clashroyale.db`.
- **Completeness**: Complete (but hardcodes `clashroyale.db` path locally).
- **Dependencies**: `sqlite3`.

### `features/feature_engine.py`
- **Purpose**: Extracts 12 engineered features from an 8-card deck.
- **Public Functions/Classes**:
  - `load_card_knowledge()` / `load_card_library()`: Parsers.
  - `FeatureEngine`: Main class. Exposes `get_card`, `get_library_card`, `get_card_name`, `get_full_card`, `build_deck`, `iter_cards`, `compute_average_elixir`, `compute_spell_count`, `compute_building_count`, `compute_has_champion`, `compute_has_big_spell`, `compute_has_small_spell`, `compute_has_evolution`, `compute_air_hitting_count`, `compute_splash_count`, `compute_win_condition_count`, `compute_durability_index`, `compute_damage_index`, and `extract_features`.
- **Completeness**: Partial (crashes on `P.E.K.K.A` naming mismatch).
- **Dependencies**: `json`, `pathlib`, `typing`.

### `features/card_categories/structural.py`
- **Purpose**: Helper properties for card entries.
- **Public Functions/Classes**: `name`, `rarity`, `elixir`, `max_level`, `has_evolution`, `is_champion`.
- **Completeness**: Complete.
- **Dependencies**: `json`, `pathlib`.

### `preprocessing/deck_features.py`
- **Purpose**: Appends deck features in batch into pandas dataframes.
- **Public Functions/Classes**: `add_deck_features`, `_build_feature_lookup`, `_merge_feature_lookup`.
- **Completeness**: Complete (blocked by `FeatureEngine` bugs).
- **Dependencies**: `json`, `pandas`, `features.feature_engine`.

### `preprocessing/preprocess.py`
- **Purpose**: Preprocessing pipeline to format decks, map archetypes, and apply target encoding.
- **Public Functions/Classes**: `generate_augmented_dataset`.
- **Completeness**: **Broken** (imports non-existent `archetype_features`, hardcodes database paths).
- **Dependencies**: `sqlite3`, `pandas`, `numpy`, `json`, `os`, `archetype_features`.

### `scrape_royale_api.py`
- **Purpose**: Web scraper that downloads popular trending decks from RoyaleAPI and outputs them to `meta_archetypes_library.json`.
- **Public Functions/Classes**: `scrape_and_merge_meta_decks`, `save_library`.
- **Completeness**: Complete (depends on live HTML structure).
- **Dependencies**: `requests`, `bs4`, `json`, `time`.

### `train_xgboost.py`
- **Purpose**: Model training script.
- **Completeness**: Partial (hardcoded weights and file paths, misplaced in root).
- **Dependencies**: `pandas`, `xgboost`, `sklearn`.

---

## 5. Feature Engineering Audit

The following table documents the 12 deck-level features computed in `FeatureEngine`:

| Feature Name | Type | Description | Computation Logic / Source | Data Source |
| :--- | :--- | :--- | :--- | :--- |
| `average_elixir` | `float` | Mean elixir cost of the 8-card deck. | Sum of card elixir costs divided by 8. | `card_library.json` |
| `spell_count` | `int` | Total number of spells in the deck. | Counts cards where `card_type` is `SPELL`. | `card_knowledge.json` |
| `building_count` | `int` | Total number of buildings in the deck. | Counts cards where `card_type` is `BUILDING`. | `card_knowledge.json` |
| `has_evolution` | `bool` | Presence of an evolvable card. | Checks if `evolution_ability` is `True` for any card. | `card_knowledge.json` |
| `has_champion` | `bool` | Presence of a Champion rarity card. | Checks if card rarity is `champion`. | `card_library.json` |
| `has_big_spell` | `bool` | Deck has a heavy damage spell. | Checks if card name is in `BIG_SPELLS` constant set. | `card_library.json` |
| `has_small_spell` | `bool` | Deck has a cheap utility spell. | Checks if card name is in `SMALL_SPELLS` constant set. | `card_library.json` |
| `air_hitting_count` | `int` | Number of cards targeting air. | Counts cards where `can_attack_air` is `True`. | `card_knowledge.json` |
| `splash_count` | `int` | Number of area-of-effect splash cards. | Counts cards where `splash_damage` is `True`. | `card_knowledge.json` |
| `win_condition_count`| `int` | Number of primary win conditions. | Counts cards where `win_condition` is `True`. | `card_knowledge.json` |
| `durability_index` | `int` | Aggregated durability rating. | Sum of `HP_SCORE` mapping of card `hp_level`. | `card_knowledge.json` |
| `damage_index` | `int` | Aggregated damage rating. | Sum of `DPS_SCORE` mapping of card `dps_level`. | `card_knowledge.json` |

### Internal Mechanics of `FeatureEngine`
Upon instantiation (`FeatureEngine()`), the class loads both `card_library.json` and `knowledge/card_knowledge.json` into memory. When extracting features via `extract_features(deck_ids)`, it maps each card ID to its name via the library, retrieves the detailed tags from the knowledge base, constructs a list of combined objects, and passes this list to the individual helper functions shown above.

---

## 6. Card Knowledge Audit

The `knowledge/card_knowledge.json` file is a structured database mapping 121 card names to strategic properties. 

### Schema Structure
Each card entry conforms to the following schema:
- `structural`: Basic attributes. Exposes `card_type`, `movement_type`, `movement_speed`, `attack_style`, and `targets`.
- `combat`: Specific combat behaviors. Exposes `hp_level`, `dps_level`, `attack_speed`, `attack_range`, `splash_damage`, `piercing_attack`, `chain_attack`, `can_attack_air`, `can_attack_ground`, `charge`, `knockback`, and `reset`.
- `abilities`: Auxiliary behaviors. Exposes `death_damage`, `spawn_damage`, `spawn_units`, `heal`, `freeze`, `reset`, `dash`, `chain_attack`, `rage_aura`, `evolution_ability`, and `unique_abilities`.
- `strategic`: Macro archetypes. Exposes `primary_role`, `secondary_roles`, `win_condition`, `support_card`, `cycle_card`, and `archetypes`.
- `tags`: List of generic strategic and behavioral labels (e.g. `["tank", "ground", "melee"]`).

### Core Assumptions
- **Tournament Standard Baseline**: All combat properties (e.g. `hp_level`, `dps_level`, range) are evaluated at the Level 11 tournament standard.
- **Spelling Mismatches**: The keys `"P.E.K.K.A."` and `"Mini P.E.K.K.A."` in the knowledge base are spelled with trailing periods. However, in `card_library.json` they are spelled `"P.E.K.K.A"` and `"Mini P.E.K.K.A"`. This assumption mismatch causes `KeyError` crashes in the pipeline.
- **Non-combatants**: Spells and static buildings assume `null` or `STATIC` movement speed and attack range parameters.

---

## 7. Current Progress

Here is the completion analysis across the subsystems:

- **Collector (90%)**: Breadth-first crawl is fully operational.
- **Database (95%)**: Fully structured schema and indexes are set up in SQLite.
- **Knowledge Base (95%)**: Standard tags are populated for all 121 cards, but naming mismatches block execution.
- **Feature Engine (70%)**: Features are defined, but blocked by the PEKKA name mismatch.
- **Deck Feature Pipeline (80%)**: Structured with pandas, but blocked by downstream dependencies.
- **Preprocessing (30%)**: Unusable due to missing module `archetype_features.py` and local db path issues.
- **Training (70%)**: Script written but misplaced in root and relies on hardcoded split weights.
- **Model Evaluation (50%)**: Basic cross-validation is set up but lacks confusion matrices, ROC/PR curves, or visualization suites.
- **Deployment (0%)**: No serving layer or automated batch inference has been built.

**Overall Project Readiness: ~65%**

---

## 8. Technical Debt

A catalog of known technical debt and incomplete components in the repository:

- **0-Byte Placeholder Files**: The following stub files exist in the repository but contain no code:
  - `collector/queue_manager.py`
  - `features/deck_similarity.py`
  - `features/generate_metadata.py`
  - `features/knowledge/import_research.py`
  - `features/card_categories/__init__.py`
  - `features/card_categories/archetypes.py`
  - `features/card_categories/combat.py`
  - `features/card_categories/strategic.py`
  - `features/card_categories/synergies.py`
- **Missing Module**:
  - `archetype_features` is referenced in `preprocess.py` but is completely absent from the codebase.
- **Misplaced Files**:
  - `train_xgboost.py` is in the root directory rather than `training/`.
  - Scraper scripts (`scrape_royale_api.py`) and various test scripts are in the root directory rather than a designated folder.
- **Redundant Files**: `features/knowledge/card_knowledge_draft.json` is a redundant exact copy of `card_knowledge.json`.
- **Hardcoded File References**:
  - `db_name = "clashroyale.db"` inside `preprocess.py` and `check_db.py` looks in the current working directory, which breaks execution when running scripts from other directories.
- **Hardcoded ML Parameters**: `scale_weight_ratio = 303.0 / 403.0` inside `train_xgboost.py` assumes a specific class imbalance ratio from a prior run, which will become inaccurate as new crawl data is added.

---

## 9. Known Bugs

| Bug Description | Severity | Affected Files | Recommended Fix |
| :--- | :--- | :--- | :--- |
| **PEKKA KeyError** | **High** | `features/knowledge/card_knowledge.json`, `features/feature_engine.py` | Remove trailing dots from `"P.E.K.K.A."` and `"Mini P.E.K.K.A."` keys inside `card_knowledge.json` to match `card_library.json`, or normalize names inside `FeatureEngine.get_card()`. |
| **Broken Preprocessing Import** | **High** | `preprocessing/preprocess.py` | Implement the missing `archetype_features.py` file containing `load_archetypes`, `add_archetype_columns`, and `matchup_win_rate_features`. |
| **Database File Path Mismatch** | **Medium**| `preprocessing/preprocess.py`, `database/check_db.py` | Import and utilize `DB_PATH` from `database.storage` (or `collector.database`) instead of using relative local string filenames. |
| **2v2 IndexError Risk** | **Medium**| `collector/parser.py` | Validate that the lists `team` and `opponent` are not empty before accessing index `0` (e.g. check length first), and skip or handle non-1v1 formats. |
| **Mismatched Test Comments** | **Low** | `test_feature_engine.py` | Update comments beside the card IDs in the test deck list (e.g., ID `26000015` represents `Baby Dragon`, not `Fireball`). |

---

## 10. Dead Code

The following files represent dead, redundant, or unused resources in the current workspace:

- `features/knowledge/card_knowledge_draft.json` (Redundant exact copy of `card_knowledge.json`). **Can be safely deleted.**
- `features/knowledge/import_research.py` (0 bytes). **Can be safely deleted.**
- `features/generate_metadata.py` (0 bytes). **Can be safely deleted.**
- `collector/queue_manager.py` (0 bytes). **Can be safely deleted** since queue operations are handled inside `database.py`.
- `features/card_categories/` 0-byte placeholders: `archetypes.py`, `combat.py`, `strategic.py`, `synergies.py`. **Can be safely deleted** if modular refactoring is abandoned, or should be filled to resolve dead import paths.

---

## 11. Architecture Review

### Strengths
- **Decoupled Collection Layer**: The crawler (`collector.py`) and API client (`api_client.py`) are highly decoupled from the machine learning model.
- **Relational Storage**: The database schema is cleanly indexed, which prevents database slowdowns as crawler runs scale.
- **Parquet Exchange format**: Using parquet files as the exchange medium between preprocessing and training separates feature extraction and modeling, allowing iterative training runs without repeating feature calculations.

### Weaknesses
- **Monolithic Feature Engine**: `feature_engine.py` contains hardcoded constants like `BIG_SPELLS` and `SMALL_SPELLS`. This violates modularity principles, as the designated files under `card_categories/` are empty.
- **Root Clutter**: Model training, scraping, and debugging scripts clutter the root folder, which reduces overall project legibility and violates maintainability standards.
- **Brittle Scraping**: The meta deck scraper (`scrape_royale_api.py`) relies on fragile HTML elements and class selectors, which can break if the external website updates.

---

## 12. Current Roadmap

The following list shows remaining development tasks in order of dependency. **Note**: These are strictly tasks required to complete the existing pipeline, and do not introduce new features.

1. **Standardize DB References**: Change path references to utilize a single database location (`data/clashroyale.db`).
2. **Fix Naming Mismatches**: Remove trailing dots in the keys `"P.E.K.K.A."` and `"Mini P.E.K.K.A."` within `card_knowledge.json`.
3. **Delete Redundant Files**: Clean up `card_knowledge_draft.json` and empty utility scripts.
4. **Implement `archetype_features.py`**: Write the missing preprocessing logic for archetype encoding and win-rate target calculations.
5. **Re-organize Workspace Folder Layout**: Move `train_xgboost.py` to `training/` and testing scripts to a designated test directory.
6. **Decouple Spells in Feature Engine**: Refactor `feature_engine.py` to load spell categories from `card_categories/structural.py` rather than hardcoding constants.
7. **Calculate Dynamic Imbalance Weights**: Update `train_xgboost.py` to calculate negative-to-positive ratio dynamically on the input dataset.
8. **Handle API 2v2 logs**: Enhance `BattleParser` to gracefully ignore or process non-1v1 battle logs without index out-of-range crashes.
9. **Implement Deck Similarity Module**: Populate `features/deck_similarity.py` using Jaccard Similarity to compare decks.
10. **Build Model Performance Metrics Visualizations**: Create confusion matrices and ROC/PR curve plots saved into `docs/` during model validation.

---

## 13. Recommended Next Steps

The next 10 development tasks, ordered by priority:

1. **Task**: Fix PEKKA name mismatch in `card_knowledge.json`.
   - **Files Affected**: `features/knowledge/card_knowledge.json`.
   - **Difficulty**: Easy.
   - **Why**: Resolves a critical bug that blocks feature extraction for any deck containing a PEKKA.
2. **Task**: Standardize SQLite path references.
   - **Files Affected**: `preprocessing/preprocess.py`, `database/check_db.py`.
   - **Difficulty**: Easy.
   - **Why**: Ensures that scripts looking for the database resolve to `data/clashroyale.db` correctly.
3. **Task**: Re-create missing `archetype_features.py` module.
   - **Files Affected**: `preprocessing/archetype_features.py` (New File).
   - **Difficulty**: Hard.
   - **Why**: Restores the broken preprocessing pipeline, enabling archetypes to be mapped.
4. **Task**: Remove redundant draft knowledge base.
   - **Files Affected**: `features/knowledge/card_knowledge_draft.json`.
   - **Difficulty**: Easy.
   - **Why**: Keeps the repository clean and prevents confusion regarding the source of truth.
5. **Task**: Move `train_xgboost.py` to `training/`.
   - **Files Affected**: `train_xgboost.py` (Path change).
   - **Difficulty**: Easy.
   - **Why**: Adheres to the defined repository structure and cleans the root folder.
6. **Task**: Dynamically compute `scale_pos_weight` in `train_xgboost.py`.
   - **Files Affected**: `train_xgboost.py`.
   - **Difficulty**: Easy.
   - **Why**: Adapts the training process to changing data dimensions as the crawler runs.
7. **Task**: Fix comments in `test_feature_engine.py`.
   - **Files Affected**: `test_feature_engine.py`.
   - **Difficulty**: Easy.
   - **Why**: Fixes misleading comments that mismatch with actual card IDs.
8. **Task**: Add length checks in `BattleParser.parse`.
   - **Files Affected**: `collector/parser.py`.
   - **Difficulty**: Medium.
   - **Why**: Prevents index crashes on non-1v1 battle logs.
9. **Task**: Populate `features/deck_similarity.py`.
   - **Files Affected**: `features/deck_similarity.py`.
   - **Difficulty**: Medium.
   - **Why**: Enables metric calculations of how close a user's deck is to meta archetypes.
10. **Task**: Implement evaluation plotting utilities.
    - **Files Affected**: `train_xgboost.py`.
    - **Difficulty**: Medium.
    - **Why**: Allows engineers to visually evaluate model precision, recall trade-offs, and ROC curves.

---

## 14. Repository Health Score

- **Architecture**: `6/10` (Modular folders are defined but contain 0-byte placeholders, and training/tests clutter the root).
- **Documentation**: `7/10` (Guidelines are informative, but the codebase lacks inline comments and API references).
- **Readability**: `8/10` (Variable naming and class structures are clear and follow standard Python guidelines).
- **Feature Engineering**: `7/10` (A comprehensive 12-feature list is defined, but it relies on hardcoded constants).
- **Data Pipeline**: `5/10` (Broken due to missing preprocessing dependencies and path mismatches).
- **Training Pipeline**: `6/10` (XGBoost code is written, but depends on a hardcoded class imbalance ratio).
- **Testing**: `5/10` (Basic scripts exist but lack assertions or integration with a test suite).
- **Maintainability**: `5/10` (Presence of 0-byte files, redundant copies, and hardcoded values decreases maintenance scores).

### **Overall Score: 6.1 / 10**

---

## 15. Final Handover

Welcome to the Clash Royale ML project! 

### Where to Start
1. Begin by fixing the naming mismatch of P.E.K.K.A. in `features/knowledge/card_knowledge.json`.
2. Implement the missing `preprocessing/archetype_features.py` file to restore the data cleaning and target encoding step.
3. Clean the repository by deleting `card_knowledge_draft.json` and moving `train_xgboost.py` to `training/`.

### Stable Components (Do NOT Modify)
- The database schema initialized by `database/storage.py` is stable and has robust indexing.
- The crawler crawl loop inside `collector/collector.py` and API client inside `api_client.py` are stable and function correctly.

### Highest-Priority Work
- Implementing the missing archetype features encoding module so that the preprocessing pipeline can run and generate the Parquet dataset required by the training script.
