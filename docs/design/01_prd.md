# Product Requirements Document (PRD)
## Clash Royale Matchup Predictor — Version 1.0.0

---

## 1. Overview & Product Goals

The Clash Royale Matchup Predictor is an interactive, single-page web application that uses machine learning to evaluate and predict win probabilities between any two Clash Royale decks under equal player skill conditions. 

In competitive Clash Royale, evaluating match outcomes from public ladders is heavily confounded by player skill differences. This product isolates pure deck-on-deck intrinsic strength, giving players, coaches, and analysts a statistical tool to evaluate deck matchup dynamics.

### Primary Objectives:
*   Provide a responsive, high-fidelity user interface where players can construct or load any two Clash Royale decks side-by-side.
*   Isolate intrinsic card interaction strength by querying a pre-trained Siamese neural network model (MatchupModel) exposed via a FastAPI backend.
*   Expose key engineered deck properties (such as elixir, splash counts, buildings, and win conditions) in a side-by-side comparison table to explain model predictions.

---

## 2. Target Audience

1.  **Competitive Players**: High-ladder and tournament players who want to test meta deck matchup advantages before choosing their decks.
2.  **Coaches & Analysts**: Esports professionals who need objective data to prepare pick-and-ban strategies or evaluate team deck lineups.
3.  **Enthusiasts**: General players seeking to understand why their custom decks struggle against popular meta archetypes.

---

## 3. Scope Boundaries & Out-of-Scope (Constraints)

To maintain a lean, high-performance product, the following features are strictly **Out of Scope** for Version 1.0.0:
*   **No User Authentication/Accounts**: The app is completely public and anonymous. No user profiles, login screens, or database associations.
*   **No Player Tag Lookups**: Users cannot enter player tags (e.g., `#CG0V8QC9J`) to import battle history or current decks.
*   **No Replay Analysis**: No video or log uploads to analyze past matches.
*   **No Deck Recommendations**: The system will not suggest counters, modifications, or replacements for selected decks.
*   **No Saved Decks**: Decks cannot be persisted on the server or in local storage (except for basic session state).
*   **No QR Export**: No generation of QR codes for deck sharing.
*   **No AI Explanations**: No natural language generation (e.g. OpenAI/Gemini explanation of results).
*   **No Chat/Social Features**: No communication features, forums, or sharing feeds.
*   **No Monetization**: No ads, subscription walls, or microtransactions.

---

## 4. Functional Requirements

### FR-1: Two-Deck Comparison Interface
*   **Description**: The interface must present two decks (designated Deck A / Left and Deck B / Right) side-by-side.
*   **Deck Slot Rules**: Each deck consists of exactly 8 card slots.
*   **Deck Actions**: Each deck must support exactly three actions:
    *   **Build**: Opens a card selector modal showing all 122 Clash Royale cards.
    *   **Copy**: Copies the deck's card IDs to the user's system clipboard as a structured JSON string or shareable array.
    *   **Paste**: Parses card IDs from the clipboard and loads them into the designated deck slot.
*   **Card Pool**: The builder must render the full, updated pool of 122 cards sourced from `features/card_library.json`.

### FR-2: Deck Library
*   **Description**: A static deck explorer panel separate from the active deck builders.
*   **Categories**: The library must display decks grouped into categories (e.g., **Meta Decks** and **Popular Decks**).
*   **Card Information**: Each deck entry must display:
    *   Deck Name (e.g. "Hog 2.6 Cycle", "PEKKA Bridge Spam").
    *   Calculated Average Elixir.
    *   Eight card images representing the deck.
    *   A **Copy** button to save the deck to the user's system clipboard.
*   **Integration**: Users manually choose where to paste copied decks (Deck A or Deck B) using the deck's respective **Paste** action.

### FR-3: Prediction Panel
*   **Description**: Displays prediction outcomes immediately when both Deck A and Deck B contain exactly 8 valid cards.
*   **Indicators**:
    *   **Favored Deck**: Visually highlights the deck with the higher predicted win probability.
    *   **Win Probability**: Renders the exact win percentage (e.g., `54.2%`) derived from the backend `/predict` endpoint.
    *   **Confidence**: Displays a categorical confidence band based on the win probability:
        *   `>= 58.0%`: Strongly Favored
        *   `52.0% - 57.9%`: Slightly Favored
        *   `48.0% - 51.9%`: Even Matchup
        *   `42.0% - 47.9%`: Slightly Unfavored
        *   `< 42.0%`: Strongly Unfavored

### FR-4: Detailed Deck Analysis Section
*   **Description**: A collapsible section located below the prediction display that details the comparative features of both decks.
*   **Engineered Features**: Features are populated by the feature engine and compared side-by-side:
    *   **Average Elixir**: Elixir cost mean.
    *   **Tankiness**: Represented by the sum of card durability levels (durability index).
    *   **Damage**: Represented by the sum of card damage/DPS levels (damage index).
    *   **Splash**: Count of splash damage cards.
    *   **Buildings**: Count of building structures.
    *   **Cycle**: Calculated average elixir cost or spell count indicator.
    *   **Air Defense**: Count of cards capable of attacking air troops.
    *   **Win Conditions**: Count of cards categorized as win conditions.

---

## 5. Non-Functional Requirements (NFRs)

*   **NFR-1: Performance & Latency**: API predictions must return in less than 100 milliseconds under ordinary network conditions.
*   **NFR-2: Mobile-First Responsiveness**: The UI must adapt seamlessly between mobile screens (stacked layouts), tablet screens (compact grid), and desktop screens (side-by-side columns).
*   **NFR-3: Accessibility (WCAG 2.1 AA)**: Color choices must satisfy contrast criteria (4.5:1 ratio for text). Focus states, screen-reader labels, and keyboard navigability are mandatory.
*   **NFR-4: Browser Compatibility**: Fully compatible with the latest stable versions of Chrome, Safari, Edge, and Firefox.

---

## 6. Open Questions
1.  **Clipboard Format**: What is the preferred format for the clipboard data? A raw JSON list of card IDs (e.g., `[26000000, 26000001, ...]`) is simple, but a custom string format or official Clash Royale deck link format might be more user-friendly.
2.  **API Fallbacks**: How should the client handle cases where the backend is unresponsive or under high load? Should it show a degraded performance warning or local fallback predictions?
3.  **Local Storage Caching**: Should the last selected decks A and B be saved in local storage to prevent loss of state on page refresh?

---

## 7. Future Improvements
1.  **Clash Royale Official Link Parsing**: Support parsing official Clash Royale deck links (e.g., `clashroyale://copyDeck?deck=...`) directly in the paste field.
2.  **Retraining Indicators**: Expose the model version and training date on the interface to assure users that predictions reflect recent game balance patches.

---

## 8. Implementation Notes
*   Ensure that the card image URLs map directly to the `icon_url` paths defined in the master `features/card_library.json` file.
*   To prevent cross-site scripting (XSS) risks, do not use `eval` or unsafe HTML parsers on clipboards during paste actions.

---

## 9. Potential Risks
*   **Card Library Desynchronization**: If new cards are added to Clash Royale, the frontend might render missing images if `card_library.json` is not updated simultaneously.
*   **Clipboard Permissions**: Modern browsers block clipboard reads without explicit user permission. The UI must handle clipboard access rejections gracefully and provide an alternative text-paste fallback field.
