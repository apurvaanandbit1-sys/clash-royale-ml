# Clash Royale Matchup Predictor

A machine learning pipeline designed to predict the win probability of Clash Royale battles using engineered deck-level features.

---

## 1. What is Clash Royale?

Clash Royale is a real-time, player-versus-player card strategy game developed by Supercell. In a match, two players face off on a dual-lane arena with the objective of destroying the opponent's towers while defending their own. Before entering combat, each player constructs a "deck" of exactly 8 cards, which represent troop units, spells, or stationary buildings. Deploying a card costs "elixir," a resource that regenerates gradually over time. 

Predicting match outcomes based on deck composition is a challenging machine learning problem. Decks exhibit complex, non-linear card synergies (e.g., combining high-health tanks with splash support) and strict rock-paper-scissors counter relationships (e.g., swarms counter tanks, but are neutralized by area-of-effect spells). A successful prediction model must capture these relationships from static card combinations.

---

## 2. Project Motivation

Typical machine learning approaches to card game matchups rely on raw card IDs as sparse, high-cardinality indicators. This forces the model to learn every mechanical property from scratch, leading to high data requirements and poor generalization to new or custom deck combinations.

This project overcomes these limitations by utilizing **engineered gameplay features**. We project the high-cardinality card matrix into a dense, interpretable feature space by mapping cards to domain-specific metadata (such as elixir costs, troop roles, splash damage capability, and durability indices). This allows our models to generalize predictions to completely unseen decks by analyzing their mechanical compositions and tactical behaviors.

---

## 3. Current Features

* **Seed-Based BFS Collector**: Traverses player logs starting from high-trophy competitive seed players to assemble a database of high-level matches.
* **Metadata-Rich SQLite Database**: Custom schema storing battle records, raw API JSON payloads, card level mappings, arena names, and collector version metadata.
* **Card Knowledge Base**: Granular combat classification mapping structural roles, hitpoints, and combat styles for all 122 playable cards (including *Ronin*).
* **Feature Engineering Engine**: Extracts 12 deck-level features (including elixir averages, spell/building ratios, evolution and champion indicators, and durability/damage indices).
* **Parquet Preprocessing**: Cleans duplicates, enforces 8-card length constraints, and compiles the SQLite records into Parquet format.
* **Automated Unit Tests**: Test suites verifying database integrity, library size constraints, and feature engineering outputs.

---

## 4. Architecture

```
   Official Clash Royale API
               │
               ▼
        Battle Collector
               │
               ▼
     SQLite Database (data/clashroyale.db)
               │
               ▼
   Preprocessing Pipeline (preprocessing/preprocess.py)
               │
               ▼
   Feature Engineering (features/feature_engine.py)
               │
               ▼
     Engineered Dataset (matches_with_features.parquet)
               │
               ▼
    Machine Learning Model (planned training pipeline)
```

---

## 5. Repository Structure

* [collector/](file:///c:/projects/clash-royale-ml/collector/): API credentials handler, JSON response parser, and breadth-first crawler loop.
* [database/](file:///c:/projects/clash-royale-ml/database/): SQLite database storage schemas and initialization scripts.
* [features/](file:///c:/projects/clash-royale-ml/features/): Card definitions registry, tactical card knowledge base, and feature engineering extraction modules.
* [preprocessing/](file:///c:/projects/clash-royale-ml/preprocessing/): Data transformation pipeline to load SQLite records and output Parquet matrices.
* [training/](file:///c:/projects/clash-royale-ml/training/): Module designated for baseline and ensemble classifier model training.
* [tests/](file:///c:/projects/clash-royale-ml/tests/): Test suites validating data storage, library lengths, and feature extraction.
* [docs/](file:///c:/projects/clash-royale-ml/docs/): Research logs, enums definitions, and project audits.

---

## 6. Technology Stack

* **Python**: Core programming language.
* **SQLite**: Transactional database warehouse.
* **Pandas**: Structured dataset transformations.
* **NumPy**: Vectorized numeric operations.
* **Scikit-learn**: Data utilities and baseline modeling.
* **Pyarrow**: High-performance Parquet file read/write operations.
* **XGBoost (planned)**: Gradient boosted trees classifier.
* **FastAPI (planned)**: Real-time prediction web API.

---

## 7. Project Status

| Component | Status | Description |
| :--- | :---: | :--- |
| **Data Collection** | ✅ Implemented | API crawler fetches log records and saves raw JSON with card levels. |
| **Knowledge Base** | ✅ Implemented | Combat classification schema mapping for all 122 playable cards. |
| **Feature Engineering** | ✅ Implemented | Extracts 12 structural/combat deck indices. |
| **Preprocessing** | ✅ Implemented | Aggregates and clean-splits database records into Parquet format. |
| **Model Training** | 🚧 In Progress | Baseline constant models and XGBoost training pipeline prototype. |
| **Evaluation Suite** | 📅 Planned | Calibration curves, Brier score mapping, and SHAP explanations. |
| **Deployment API** | 📅 Planned | FastAPI microservice exposing matchup probability endpoints. |

---

## 8. Development Roadmap

### Completed
- [x] Create seed-based BFS player crawler.
- [x] Design SQLite schema storing card levels and raw JSON.
- [x] Build card library with official API metadata.
- [x] Classify strategic combat tags for all 122 cards.
- [x] Integrate support for *Ronin* in card library and knowledge base.
- [x] Implement 12-dimensional engineered deck features.
- [x] Implement Parquet dataset preprocessing.

### Current
- [/] Move training scripts from root prototype to `training/` package.
- [/] Scale crawler database from 8 battles to $\ge$ 5,000 battles.

### Future
- [ ] Implement leak-proof chronological splits.
- [ ] Implement symmetric dataset augmentation.
- [ ] Calibrate prediction probabilities using Isotonic regression.
- [ ] Compute model explainability metrics via SHAP.
- [ ] Deploy prediction backend via FastAPI.

---

## 9. Installation

Clone the repository and install the dependencies:

```bash
# Clone the repository
git clone https://github.com/yourusername/clash-royale-ml.git
cd clash-royale-ml

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
.venv\Scripts\activate     # On Windows

# Install required dependencies
pip install -r requirements.txt

# Create a local environment file (.env) and add your credentials:
# CR_API_KEY=your_official_clash_royale_api_developer_key
```

---

## 10. Running the Project

### Collect Battles
To run the database crawler and populate the SQLite database:
```bash
python collector/collector.py
```

### Preprocess Data
To load matches from the database, apply feature engineering, and output Parquet files:
```bash
python -m preprocessing.preprocess
```

### Run Tests
To run unit tests verifying database, library, and feature extraction states:
```bash
python tests/test_database.py
python -m tests.test_card_library
python tests/test_feature_engine.py
```

---

## 11. Future Work

* **Symmetric Augmentation**: Mirror matches (`A vs B` and `B vs A`) to double training size and ensure prediction symmetry.
* **Leak-Proof Splitting**: Enforce group-aware chronological validation to prevent leakage of identical matchups across splits.
* **Matchup-Aware Interactions**: Add feature interactions capturing specific counter relationships (e.g. spell-to-troop ratios).
* **Probability Calibration**: Calibrate model probabilities using Platt scaling or Isotonic regression to align outputs with real-world win rates.
* **Model Explainability**: Add SHAP force plots to interpret feature contributions per prediction.

---

## 12. Acknowledgements

* **Official Clash Royale API**: Source of raw battle logs.
* **Supercell**: Developers of Clash Royale.
* **RoyaleAPI**: Taxonomy and community card metadata reference.

*Disclaimer: This is an independent educational/research project and is not affiliated with, endorsed by, or associated with Supercell. Clash Royale assets and trademarks are properties of Supercell.*
