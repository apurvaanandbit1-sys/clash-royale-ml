# Clash Royale ML - Project Status

## 1. Project Objective
The goal of this machine learning project is to accurately predict the outcomes of Clash Royale battles. By utilizing historical match data and advanced feature engineering, the model seeks to identify winning patterns based on deck compositions, structural archetypes, and card synergies.

## 2. Current Architecture
The architecture follows a linear pipeline:
Data is first pulled from the official Clash Royale API using Python modules in the `collector/` directory and stored in a SQLite database (`database/`). From there, the raw data is cleaned and augmented with macro-archetype mappings via scripts in the `preprocessing/` directory, outputting a consolidated `matches_with_archetypes.parquet` file. Finally, an XGBoost classifier reads this matrix to train and evaluate a predictive model in `train_xgboost.py`.

## 3. Repository Structure
- **collector/**: Houses scripts to interface with the Clash Royale API, fetch player battle logs, parse JSON responses, and manage the crawling queue.
- **config/**: Contains basic configuration parameters (e.g., crawler depth, request delays, thresholds).
- **database/**: Implements the SQLite database wrapper for persistent storage of players, queues, and match history.
- **docs/**: Stores research documents, methodology files, and project status documentation.
- **features/**: Dedicated to housing the card metadata library, the structured card knowledge base, and archetype definitions used for downstream feature engineering.
- **preprocessing/**: Contains scripts to normalize raw database inputs and augment them with engineered target and archetype features.

## 4. Completed Components
- **API Client (`collector/api_client.py`)**: Fetches data robustly from the Clash Royale API with automatic retry logic for rate limits.
- **Crawler/Orchestrator (`collector/collector.py`)**: Continuously loops to crawl the API, process matches, and dynamically add new opponents to the database queue.
- **Database Manager (`collector/database.py`, `database/storage.py`)**: Creates database schemas and handles `INSERT OR IGNORE` transactions for battles and players.
- **Battle Parser (`collector/parser.py`)**: Extracts and flattens JSON payloads from the API into deterministic, structured database rows.
- **Preprocessor (`preprocessing/preprocess.py`)**: Loads SQLite data, transforms deck strings, maps macro-archetypes, computes Bayesian-smoothed matchup win-rates, and outputs a Parquet matrix.
- **Scraper (`scrape_royale_api.py`)**: Fetches metadata and archetype mapping from RoyaleAPI.
- **Model Trainer (`train_xgboost.py`)**: Loads the Parquet matrix, drops raw text features, applies `scale_pos_weight`, trains an XGBoost classifier, and evaluates it using Stratified K-Fold CV.

## 5. Knowledge Base Status
The project separates general API card data from deeply researched combat attributes.
- **`features/card_library.json`**: An automatically generated library containing basic statistics (elixir cost, rarity, IDs, icons) for all **121** cards currently available in the API.
- **`features/knowledge/card_knowledge.json`**: A highly granular, manually curated ML knowledge base containing deep structural, combat, ability, and strategic role parameters. There are currently **16** fully mapped cards in this schema.
- **Schema Design**: The knowledge schema (`enums.md`) standardizes attributes like HP_LEVEL, PRIMARY_ROLE, and ATTACK_SPEED into strict categorical Enums, ensuring the ML pipeline ingests clean, non-hallucinated data.

## 6. Feature Engineering Status
The central `features/feature_engine.py` file is currently **empty** (not implemented). The project currently relies entirely on `preprocessing/preprocess.py` and `features/card_categories/` helper scripts for its engineered features. Existing implemented features include mapping raw deck lists to macro-archetypes using predefined `meta_archetypes_library_expanded.json` dictionaries and generating Bayesian-smoothed target win rates.

## 7. Machine Learning Pipeline Status
- **Data Collection (Crawler)**: Completed
- **Data Preprocessing & Archetype Mapping**: Completed
- **Target Encoding**: Completed
- **Model Training (XGBoost Baseline)**: Completed
- **Model Evaluation (Stratified K-Fold)**: Completed
- **Feature Engine (`feature_engine.py`)**: Not Started
- **Model Inference API/Web Service**: Not Started
- **Hyperparameter Tuning**: Not Started

## 8. Major Design Decisions
- **Separation of Card Data**: Basic properties are dynamically scraped (`card_library.json`), while nuanced game mechanics and roles are strictly managed in a separate file (`card_knowledge.json`) to prevent API changes from poisoning the ML model's understanding of card mechanics.
- **Modular Feature Engineering**: Features are meant to be structured into distinct combat, structural, and strategic domains (as evidenced by the `features/card_categories/` directory setup) rather than flattened into a massive spaghetti script.
- **Reusable Preprocessing**: The project strictly utilizes Python script pipelines (`preprocess.py`, `train_xgboost.py`) over Jupyter Notebooks, promoting a production-ready, repeatable architecture.

## 9. Git Milestones
- `391376d Add knowledge base schema`
- `f3711ff Add Gemini research batch 1`
- `6579fc8 Remove secrets and cache files`

## 10. Current TODO List
- Fully implement `features/feature_engine.py` to bridge the gap between `card_knowledge.json` properties and the model input matrix.
- Map the remaining 105 cards into the `card_knowledge.json` schema.
- Expand `database/create_tables.py` and modularize schema initialization out of `collector/database.py`.
- Handle 2v2 modes and edge cases gracefully in `BattleParser`.

## 11. Long-Term Roadmap
- **Phase 1**: Fully populate the ML Knowledge Base with all 121 cards based on the strict schema.
- **Phase 2**: Build out the `FeatureEngine` to translate granular card properties (like `hp_level` or `splash_damage`) into aggregate numerical deck features.
- **Phase 3**: Conduct deep hyperparameter tuning on the XGBoost classifier utilizing the expanded feature set.
- **Phase 4**: Deploy the trained model via a REST API to predict live matches.

## 12. Resume Value
This repository demonstrates strong fundamentals in end-to-end data engineering and applied machine learning. It highlights skills in robust API crawling (handling rate-limits), relational database management (SQLite), data cleaning/preprocessing (Pandas), schema design, categorical encoding (Bayesian smoothing), and baseline model training (XGBoost). Furthermore, it showcases strict architectural discipline by separating raw data generation from curated domain knowledge and utilizing production-ready Python scripts instead of relying solely on exploratory Jupyter Notebooks.
