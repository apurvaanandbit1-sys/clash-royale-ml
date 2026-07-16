# Clash Royale ML

Predicting Clash Royale match outcomes using engineered gameplay features and machine learning.

---

### ⚠️ Active Development

Current implementation status:

*   ✅ Data Collection
*   ✅ Knowledge Base
*   ✅ Feature Engineering
*   ✅ Preprocessing
*   🚧 Model Training
*   🚧 Evaluation
*   📅 Deployment

---

## 1. What is Clash Royale?

Clash Royale is a real-time, player-versus-player card strategy game developed by Supercell. In a match, two players face off on a dual-lane arena with the objective of destroying the opponent's towers while defending their own. Before entering combat, each player constructs a "deck" of exactly 8 cards, which represent troop units, spells, or stationary buildings. Deploying a card costs "elixir," a resource that regenerates gradually over time. 

Predicting match outcomes based on deck composition is a challenging machine learning problem. Decks exhibit complex, non-linear card synergies (e.g., combining high-health tanks with splash support) and strict rock-paper-scissors counter relationships (e.g., swarms counter tanks, but are neutralized by area-of-effect spells). A successful prediction model must capture these relationships from static card combinations.

---

## 2. Project Goals

The long-term objectives of this project are:
*   **Predict Win Probability**: Calculate the probability that a player's deck beats an opponent's deck.
*   **Generalize to Unseen Combinations**: Predict outcomes of custom or new decks that have never been seen in the training set.
*   **Utilize Engineered Gameplay Features**: Project card IDs into a dense, interpretable feature space representing mechanical properties (elixir costs, troop roles, splash damage capability) instead of using sparse high-cardinality indicators.
*   **Produce Interpretable ML Predictions**: Provide clear feature importance breakdowns showing *why* a particular matchup outcome is predicted.

---

## 3. ML Prediction Objective

```
Input:
  - Player Deck:   8 cards (e.g., Hog Cycle)
  - Opponent Deck: 8 cards (e.g., Splashyard)

Output:
  - Probability(Player Wins) (e.g., 61.8%*)
```
*\*Note: This is an illustrative example of model output, not an actual prediction.*

---

## 4. Current Dataset Status

*   **Battles Collected**: 8 battles
*   **Purpose**: Pipeline verification and validation (V&V)
*   **Target Database**: 100,000+ battles

The pipeline currently runs on a small verified dataset of competitive matches from players with $\ge 8,500$ trophies. Once model training is initialized, the collector will be run to scale the dataset.

---

## 5. Architecture

```
         Official Clash Royale API
                     │
                     ▼
              Battle Collector
                     │
                     ▼
               SQLite Database
                     │
                     ▼
            Card Knowledge Base
                     │
                     ▼
            Feature Engineering
                     │
                     ▼
            Preprocessing Loop
                     │
                     ▼
             Training Dataset
                     │
                     ▼
          Machine Learning Model
                     │
                     ▼
         Prediction API (planned)
```

---

## 6. Repository Structure

*   `collector/`: API client, JSON log parser, and BFS crawler.
*   `database/`: SQLite table setup and database storage schemas.
*   `features/`: Card definitions, strategic gameplay knowledge base, and feature extraction.
*   `preprocessing/`: Preprocessing loop that cleans duplicates and outputs Parquet datasets.
*   `training/`: Dedicated module for baseline models and ensemble classifiers.
*   `tests/`: Unit test suite verifying databases, libraries, and feature outputs.
*   `docs/`: Analysis reports, gameplay enums, and project audits.

---

## 7. Technology Stack

*   **Python**: Core programming language.
*   **SQLite**: Warehouse database.
*   **Pandas**: Dataset matrix manipulation.
*   **NumPy**: Array mathematics.
*   **Scikit-learn**: Data utilities and baseline models.
*   **Pyarrow**: High-performance Parquet format read/write.
*   **XGBoost (planned)**: Gradient boosted trees.
*   **FastAPI (planned)**: Microservice web API.

---

## 8. Project Status

| Component | Status | Description |
| :--- | :---: | :--- |
| **Data Collection** | Implemented | BFS crawler writes battle details, raw JSON, and card levels. |
| **Knowledge Base** | Implemented | Tactical maps and role tags defined for all 122 cards (including *Ronin*). |
| **Feature Engineering** | Implemented | Computes average elixir, spells/buildings counts, and durability/damage indices. |
| **Preprocessing** | Implemented | Cleans duplicates and transforms SQLite rows into Parquet matrices. |
| **Model Training** | In Progress | Baseline constant models and XGBoost prototype classifier. |
| **Evaluation Suite** | Planned | Brier score calculation, calibration plots, and SHAP visualizations. |
| **Deployment API** | Planned | Microservice backend exposing predict endpoints. |

---

## 9. Development Roadmap

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

## 10. Installation

Clone the repository and install dependencies:

```bash
# Clone the repository
git clone https://github.com/apurvaanandbit1-sys/clash-royale-ml.git
cd clash-royale-ml

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
.venv\Scripts\activate     # On Windows

# Install required dependencies
pip install -r requirements.txt

# Create a local .env file in the root directory:
# CR_API_KEY=your_clash_royale_api_developer_key
```

---

## 11. Running the Project

### Collect Battles
Run the crawler to pull battle logs and populate the SQLite database:
```bash
python collector/collector.py
```

### Preprocess Data
Run the preprocessing script to clean matches, extract engineered features, and export Parquet files:
```bash
python -m preprocessing.preprocess
```

### Run Tests
Execute the unit tests to check database connections, card library lengths, and feature extractions:
```bash
python tests/test_database.py
python -m tests.test_card_library
python tests/test_feature_engine.py
```

---

## 12. Future Improvements

*   **Symmetric Augmentation**: Mirror decks (`A vs B` and `B vs A`) to double dataset volume and ensure prediction symmetry.
*   **Leak-Proof Splitting**: Group-aware chronological validation splits to prevent match pair leaks.
*   **Matchup-Aware Interactions**: Add feature interactions targeting specific counter rules (e.g. spell-to-troop counter indices).
*   **Probability Calibration**: Apply Platt scaling or Isotonic regression to align raw logits with empirical win rates.
*   **Model Explainability**: Add SHAP summary plots to show feature impact on predicted outcomes.
*   **FastAPI Deployment**: Expose predicted win probability endpoints.

---

## 13. License

License to be added.

---

## 14. Acknowledgements

*   **Official Clash Royale API**: Data source for battle log logs.
*   **Supercell**: Developer of Clash Royale.
*   **RoyaleAPI**: Taxonomy and community metadata reference.

*Disclaimer: This is an independent educational/research project and is not affiliated with, endorsed by, or associated with Supercell. Clash Royale assets and trademarks are properties of Supercell.*
