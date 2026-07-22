# Version Roadmap
## Clash Royale Matchup Predictor — Version 1.0.0

---

## 1. Roadmap Overview

This roadmap defines the structured release plan for the Clash Royale Matchup Predictor. It outlines features grouped across three milestones (V1, V2, and V3), aligning frontend layouts with the existing ML research outcomes and planned system expansions.

---

## 2. Milestone Details

### Version 1.0.0 — Production Baseline (Current State)
*   **Backend Model**: Production `MatchupModel` Siamese neural network utilizing Deep Sets average-pooled embeddings and Bradley-Terry skew-symmetric comparison head.
*   **Serving Runtime**: FastAPI web server hosting local ONNX runtime sessions, exposing `/predict`, `/embeddings`, and `/health` REST API endpoints.
*   **Symmetry Assurances**: Production pipeline guarantees that self-matchups always equal 50.0% win probability and swap matchups invert targets perfectly by design.
*   **Player Skill Un-confounding**: Relative skill difference modeled during training via an odd bias function, and zeroed out at prediction (`trophy_diff = 0.0`) to isolate intrinsic matchup quality.
*   **UI Features**:
    *   Side-by-side Deck Comparison UI (Deck A and Deck B columns).
    *   Three deck slot actions: Build (Card Picker Modal), Copy, and Paste.
    *   Central Prediction Panel showing Favored Deck indicators, Win Probability, and Confidence categorical bands.
    *   Collapsible Detailed Analysis Panel comparing 12 engineered features (Average Elixir, Durability, DPS, Spells, Win Conditions, Splash, etc.).
    *   Deck Library tabs supporting Meta and Popular Decks templates with Copy buttons.

### Version 2.0.0 — Interactive Explorer & Suggestions (Intermediate Phase)
*   **Backend Enhancements**:
    *   Expose `/embeddings` coordinate metadata to serve 2D card layout coordinate systems.
    *   Cache popular matchup queries to accelerate server predictions.
*   **UI Features**:
    *   **2D Card Embeddings Visualizer**: Interactive t-SNE coordinate chart mapping all 122 Clash Royale cards. Hovering nodes highlights neighboring cards with similar learned representation properties (e.g. showing win conditions grouped together).
    *   **Offline Session Recovery**: Uses browser local storage to preserve chosen compare decks so they persist across page refreshes.
    *   **Clash Royale Link Parsing**: Parse official Clash Royale copy-deck URLs in the deck paste input field to simplify mobile import flows.

### Version 3.0.0 — Deep Optimization & Graph Representations (Advanced Phase)
*   **Backend Model Retraining**:
    *   Upgrade the encoder from Deep Sets average pooling to a Graph Neural Network (GNN) or transformer architecture to capture card cycle speeds and play order.
*   **UI Features**:
    *   **Deck Optimizer Component**: Evaluates a deck and recommends card replacements to maximize win probability against target meta archetypes.
    *   **Real-time Counter suggestions**: Lists top counters to the opponent's deck from the card library when the user builds a deck.
    *   **Player Tag Sync**: Connects to the Clash Royale public API to import active user decks from a player tag.

---

## 3. Open Questions
1.  **GNN Data Requirements**: Will a GNN deck representation require collecting play-order telemetry data from live matches (which is not available in standard battle logs)?
2.  **API Key Constraints**: Direct player tag sync requires official API key credentials. Should the backend proxy these requests to prevent exposing credentials on the client-side?

---

## 4. Future Improvements
1.  **Self-hosted t-SNE calculation**: Allow the backend to recalculate t-SNE coordinates dynamically if new card embeddings are trained.
2.  **Automated balance updates sync**: Schedule weekly checks against RoyaleAPI to fetch trending deck list templates and update the static library.

---

## 5. Implementation Notes
*   Milestone V2 utilizes the existing `/embeddings` endpoint which returns pre-computed t-SNE coordinates, avoiding client-side layout calculation overhead.
*   Keep V1 design components modular to allow simple expansion into V2/V3 features without requiring layout refactoring.

---

## 6. Potential Risks
*   **Rotational Embeddings Drift**: Embeddings trained under new seeds will drift coordinate positions. The 2D embeddings visualizer in V2 must map specifically to coordinates outputted by the current checkpoint weights.
*   **API Limits**: Player tag deck lookups in V3 will trigger external API queries that might fail or face rate limit rejections. Adequate failover alerts are required in the user flows.
