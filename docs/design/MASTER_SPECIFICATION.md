# Clash Royale Matchup Predictor — Master Specification
## Version 1.0.0 — Canonical Product & Engineering Specification

---

## 1. Executive Summary

This Master Specification is the single source of truth for the Clash Royale Matchup Predictor frontend client and backend integrations. It consolidates all previous product requirements, design guidelines, component catalog entries, user journey flows, and architectural reviews into a unified engineering blueprint. 

By defining all UI layouts, interaction parameters, API contracts, state structures, and asset requirements in this document, we ensure that senior developers and AI coding assistants can execute the implementation of Version 1.0.0 without design ambiguity or structural rework.

---

## 2. Vision

The Clash Royale Matchup Predictor serves as a premium, highly immersive companion utility for Clash Royale players, coaches, and content creators. It provides users with instant, mathematically consistent deck-on-deck win predictions under equal player skill conditions. 

The visual layout and interactive workflows are designed to feel like an authentic extension of Clash Royale itself, maintaining visual familiarity while remaining legally distinct from the official game.

---

## 3. Product Goals

*   **Isolate Intrinsic Deck Matchups**: Strip away player skill confounders (such as trophy levels) during prediction to evaluate pure card interaction dynamics.
*   **Deliver Low-Latency Predictions**: Query and display Siamese neural network outputs (MatchupModel) in less than 100 milliseconds.
*   **Establish Cognitive Familiarity**: Guarantee that Clash Royale players immediately understand how to build decks and interpret outcomes using game-inspired visual elements.
*   **Secure Automatic Extensibility**: Build a metadata-driven card catalog architecture that supports future game updates (such as the release of new card archetypes like `Ronin`) without requiring client-side code modification.

---

## 4. Target Users

1.  **Casual Competitors**: Active ladder players looking to verify if their custom decks are structurally disadvantaged against trending meta decks.
2.  **Tournament Teams & Coaches**: Professional analysts who draft deck lineups and evaluate counter matchups for competitive pick-and-ban screens.
3.  **Hiring Managers & Reviewers**: Technical evaluators reviewing this repository as a portfolio project, requiring a clean, robust, and well-documented architecture.

---

## 5. Version Scope

### Included in Version 1.0.0:
*   Side-by-side comparison workspace of exactly two decks (Deck A / Left and Deck B / Right).
*   Interactive deck builders displaying the full pool of 122 Clash Royale cards.
*   Three core deck actions: **Build** (modal selector), **Copy** (system clipboard save), and **Paste** (system clipboard read and validate).
*   Static Deck Library containing Meta and Popular Decks templates with one-click copy buttons.
*   Minimal Prediction Panel displaying only the Favored Deck name, Win Probability percentage, and categorical Confidence band.
*   Collapsible Detailed Analysis Panel comparing engineered card metrics (Elixir, Spells, Buildings, Air Defense, Win Conditions, Splash, Tankiness/Durability, and Damage/DPS) using visual progress bars and cards instead of raw text tables.

### Excluded from Version 1.0.0 (Strict Scope Boundaries):
*   No user account creation, login, or profiles.
*   No player tag searches (e.g. importing active user decks from player tags).
*   No replay analysis or match log uploading.
*   No active card recommendations (e.g., suggesting counters or slot replacements).
*   No backend database saving of custom player decks.
*   No QR code exports.
*   No Natural Language AI explanations (e.g., ChatGPT/Gemini breakdown of results).
*   No chat, forums, or social features.

---

## 6. Design Philosophy

### "Familiarity Over Novelty"
The interface must prioritize layout patterns, colors, and asset styles that Clash Royale players already know. 
*   **Action over reading**: The interface should use immediate visual card recognition rather than forcing users to read text lists.
*   **No Generic SaaS Elements**: Avoid flat, sterile data dashboards, plain grey inputs, and generic business chart libraries. The application must look and feel like a game companion.
*   **Progressive Disclosure**: Show only the essential prediction figures initially. Hide complex engineered card variables behind a collapsible section to prevent cognitive overload for casual users.

---

## 7. Core Features

*   **Side-by-Side Builder Grid**: Two identical columns housing 8 slots each, styled like the in-game battle deck editor.
*   **Sequential Slot Select Modal**: Opens the card pool overlay. Tapping card thumbnails populates slots sequentially without closing the modal after each click, cutting selection taps in half.
*   **Unified Library Cards**: Pre-configured deck lists grouped under Meta and Popular tabs, displaying miniature copies of card artwork and copy actions.
*   **FastAPI ONNX Predictor**: Instant, server-side prediction execution that handles skill un-confounding by setting `trophy_diff = 0.0` at evaluation.

---

## 8. User Stories

*   **US-1: Custom Matchup Comparison**: As a competitive ladder player, I want to manually input my deck against a popular meta deck so that I can see my win probability under equal skill conditions.
*   **US-2: Fast Meta Deck Copy**: As a tournament analyst, I want to copy a pre-built meta deck from the Library with one click and paste it directly into my comparison workspace.
*   **US-3: Sharing Decks**: As a player, I want to click copy on my custom deck configuration so that I can send the serialized card string to a friend.
*   **US-4: Visual Metric Audits**: As a coach, I want to expand the detailed deck analysis and see comparative bars of deck features (like splash cards count) to understand why a matchup is favored.

---

## 9. Complete User Journey

1.  **Entry**: User arrives on the Landing Page. The header title and a brief visual description explain the application's purpose.
2.  **Deck A Selection**: The user clicks the first empty slot in Deck A. The `CardGridPicker` modal overlay launches. The user taps 8 card thumbnails; slots are filled sequentially.
3.  **Deck B Library Load**: The user goes to the Deck Library, clicks "Copy" on the "Hog 2.6 Cycle" deck, scrolls up, and clicks "Paste" on Deck B.
4.  **Prediction Display**: The Prediction Panel instantly populates, showing "DECK B SLIGHTLY FAVORED (54.2% Win Probability / High Calibration Confidence)".
5.  **Detailed Metric Review**: The user expands the "Detailed Matchup Breakdown" panel to compare average elixir, building counts, and tankiness bars.

---

## 10. Complete UI Walkthrough

```
+-------------------------------------------------------------+
|                           HEADER                            |
|        [Clash Royale Intrinsic Matchup Predictor]           |
|                "Isolating pure deck strength"               |
+-------------------------------------------------------------+
|                                                             |
|   +-----------------------+     +-----------------------+   |
|   |        DECK A         |     |        DECK B         |   |
|   | [Build] [Copy] [Paste]|     | [Build] [Copy] [Paste]|   |
|   |  +---+ +---+ +---+    |     |  +---+ +---+ +---+    |   |
|   |  |   | |   | |   |    |     |  |   | |   | |   |    |   |
|   |  +---+ +---+ +---+    |     |  +---+ +---+ +---+    |   |
|   +-----------------------+     +-----------------------+   |
|                                                             |
|                     PREDICTION PANEL                        |
|              +----------------------------+                 |
|              |      DECK A FAVORED        |                 |
|              |     58.12% Win Rate        |                 |
|              |   (Strongly Favored)       |                 |
|              +----------------------------+                 |
|                                                             |
|                DETAILEDbreakdown (COLLAPSED)                |
|                    [ Show Comparison ]                      |
|                                                             |
|                        DECK LIBRARY                         |
|              +----------------------------+                 |
|              | [Meta Decks] [Popular]     |                 |
|              |   Hog 2.6 Cycle     [Copy] |                 |
|              |   PEKKA Bridge Spam [Copy] |                 |
|              +----------------------------+                 |
+-------------------------------------------------------------+
```

---

## 11. Visual Design Language

*   **Aesthetic Theme**: High-fidelity dark mode with rich wood/stone card containers, gold border highlights, and crisp visual separations.
*   **Layout Style**: Semi-flat borders, clean glass containers, and glowing status indicators. No distracting particle effects or heavy background loops.
*   **Visual Rhythm**: Alternating dark surface panels with clear margins to guide the eye directly to the center comparison area.

---

## 12. Design System

### Typography
*   **Font Family**: `Outfit` (Primary Sans-serif, loaded from Google Fonts).
*   **Headers**: Bold (700) weight, with size increments scaling from 2.0rem (H1) down to 1.125rem (H3).
*   **Body**: Regular (400) weight, 14px size, line-height 1.5.

### Color Tokens
*   `--bg-base`: `#0b0c10` ( Obsidian Dark Base)
*   `--bg-surface`: `#1f2833` (Deep Slate Card background)
*   `--text-primary`: `#ffffff` (Pure White)
*   `--text-secondary`: `#c5c6c7` (Muted Gray)
*   `--status-favored`: `#45f3ff` (Electric Neon Cyan/Green for positive indicators)
*   `--status-unfavored`: `#ff2e93` (Deep Pink/Red for negative indicators)
*   `--rarity-legendary`: `#ff9f1c` ( Legendary Gold)

### Elevation & Shadows
*   **Elevated Card Slots**: `box-shadow: 0 4px 6px -1px rgba(0,0,0,0.5)`.
*   **Modal Selection Menu**: `box-shadow: 0 20px 25px -5px rgba(0,0,0,0.8)`.

---

## 13. Component Catalog

### 1. `DeckBuilderSlot`
*   **Prop Interface**:
    ```typescript
    interface DeckBuilderSlotProps {
      cardId: number | null;
      slotIndex: number;
      onClick: (slotIndex: number) => void;
      onDelete: (slotIndex: number) => void;
      state: 'default' | 'hover' | 'focused' | 'loading';
    }
    ```
*   **Visual States**: Render blank placeholder if `cardId` is null. Render card artwork and gold rarity frame if populated.

### 2. `CardGridPicker`
*   **Prop Interface**:
    ```typescript
    interface CardGridPickerProps {
      isOpen: bool;
      disabledCardIds: number[]; // Disables cards already loaded in deck to prevent duplicates
      onSelect: (cardId: number) => void;
      onClose: () => void;
    }
    ```
*   **Visual States**: Display search bar at the top, followed by scrollable grid of 122 cards. Already selected cards are visually greyed out (`opacity: 0.4`, `pointer-events: none`).

### 3. `PredictionPanel`
*   **Prop Interface**:
    ```typescript
    interface PredictionPanelProps {
      winProbability: number | null;
      categoricalBand: string | null;
      isLoading: boolean;
    }
    ```

---

## 14. Interaction Patterns

### Mobile Touch Gestures
*   **No Hover Dependencies**: Every deletion or replacement action must be click/tap triggered. Hover-only overlays are strictly prohibited.
*   **Tap Actions for Cards**: Tapping a populated card slot in mobile view brings up a small, non-obtrusive bottom modal: `[Change Card] [Clear Slot] [Cancel]`.

### Sequential Slot Selection
*   When a user opens the `CardGridPicker` modal, clicking a card icon loads it into the active slot and automatically advances to the next empty slot, keeping the modal open. The modal only closes when the user clicks the close (x) button or clicks outside the modal area.

---

## 15. Animation Guidelines

*   **Card Flip Transition**: Trigger a 3D horizontal flip animation (`duration: 250ms`, `easing: ease-out`) when a card slot updates from empty to filled.
*   **Radial Glow**: Highlight the favored deck with a slow, pulsing border glow (`duration: 2000ms`, linear infinite).
*   **Scale Zoom on Selection**: Gently elevate the card scale (`transform: scale(1.04)`) when hovered or active on desktop.

---

## 16. Card Asset Guidelines

### Strict Art Rules:
*   **Official Assets Only**: The application must display the official, uncropped Clash Royale card illustrations. Simplified illustrations, flat vector representations, or third-party placeholder icons are strictly banned.
*   **Rarity Frames**: Each card thumbnail must render its official border frame (Common, Rare, Epic, Legendary, Champion, Evolution).
*   **Consistency**: The exact same image files must be shared between the Selection Menu, Deck Slots, Library Rows, and Predictions. Only the width/height CSS boundaries scale.

---

## 17. Mobile Experience

*   **Ergonomics**: Mobile buttons must have touch sizes of at least `48px` to prevent mistaps.
*   **Vertical Stacking Layout**: On screens narrower than 768px, columns stack vertically. Deck A is displayed at the top, followed by the Prediction Panel, Deck B, the Analysis Table, and the Deck Library at the bottom.

---

## 18. Accessibility (WCAG 2.1 AA)

*   **Keyboard Navigation**: Tab indexing must cycle sequentially through card slots, action buttons, and card picker items. Modals must trap focus when open.
*   **Screen Readers**: Every card and slot must have screen reader labels:
    *   Empty Slot: `aria-label="Empty Slot. Click to choose card."`
    *   Filled Slot: `aria-label="Slot showing Knight, Elixir cost 3."`
*   **Contrast Safeguards**: Text over colors must use high contrast (e.g., pure white text on deep pink/cyan background elements).

---

## 19. Information Hierarchy

*   **Primary Focal Zone**: The top-center of the screen houses the Comparison board.
*   **Secondary Focal Zone**: The Prediction Panel acts as the visual bridge between the two decks.
*   **Tertiary Details**: Comparison progress bars and library cards are styled with lower visual saturation (e.g. using muted grays) to keep focus on the active builds.

---

## 20. Prediction Experience

The prediction layout is kept strictly minimal:
*   **Favored Header**: Large, bold label: "DECK A FAVORED" or "DECK B FAVORED".
*   **Win Probability**: Large central percentage text (e.g., `58.12%`).
*   **Confidence Badge**: Colored indicator tag displaying the categorical band derived from `api/app.py`.
*   Everything else is hidden behind the Collapsible Analysis Section.

---

## 21. Detailed Analysis Design

### Progress Bars and Visual Cards (No Plain Tables)
*   Engineered deck features must be compared using visual comparison cards:
    *   **Average Elixir**: Displays a numeric comparative scale.
    *   **Durability/DPS**: Measured via relative horizontal progress bars growing from the center to indicate strength differences.
    *   **Spell / Building / Air Defense counts**: Represented by rows of small icons indicating quantities.

---

## 22. Deck Library Design

*   **Layout**: Displays categories inside a tabbed menu.
*   **Decks list**: Displays rows of miniature cards, the calculated average elixir, and a copy button.
*   **Pasting Actions**: Clicking "Copy" writes the card IDs list to the clipboard. The user manually chooses where to load it by clicking "Paste" on Deck A or Deck B action bars.

---

## 23. Technical Architecture

*   **API Service Layer**: Exposes predictions, embeddings, and health states.
*   **ONNX Inference**: The Python backend loads `matchup_model.onnx` into memory at startup. Prediction requests run inference on input arrays and return probabilities.

---

## 24. Frontend Architecture

*   **Build Tool**: Vite + React or vanilla compilation.
*   **Styling**: Vanilla CSS for maximum performance and layout flexibility.
*   **State Management**: Light client-side store (e.g. Zustand or React Context) managing the active decks list.

---

## 25. Backend Integration

The frontend connects directly to the FastAPI backend over HTTP:
```
[Client App] ───► POST /predict (req) ───► [FastAPI app.py] ───► [ONNX runtime session]
```

---

## 26. API Contracts

### `/predict` (HTTP POST)
*   **Request Model**:
    ```typescript
    interface PredictRequest {
      p1_deck: number[]; // Exactly 8 card IDs
      p2_deck: number[]; // Exactly 8 card IDs
      trophy_diff: number; // Set to 0.0 by client for intrinsic matchup
    }
    ```
*   **Response Model**:
    ```typescript
    interface PredictResponse {
      win_probability: number;
      categorical_band: string; // "Strongly Favored" etc.
      cached: boolean;
      model_version: string;
    }
    ```

### `/embeddings` (HTTP GET)
*   **Response Model**:
    ```typescript
    interface EmbeddingsResponse {
      checkpoint: string;
      projection_method: string;
      num_cards: number;
      coordinates: Record<string, [number, number]>;
    }
    ```

---

## 27. Folder Structure

```
/src
  ├── components/
  │     ├── DeckCompare/
  │     │     ├── DeckSlot.css
  │     │     └── DeckSlot.jsx
  │     ├── CardPicker/
  │     └── Prediction/
  ├── services/
  ├── store/
  └── App.jsx
```

---

## 28. State Management

The Zustand store configuration manages the active deck list state and updates predictions dynamically when both decks contain 8 card IDs:
```javascript
import create from 'zustand';

export const useDeckStore = create((set, get) => ({
  deckA: Array(8).fill(null),
  deckB: Array(8).fill(null),
  prediction: { probability: null, band: null, loading: false },
  
  setCard: (deckName, index, cardId) => {
    set(state => {
      const targetDeck = deckName === 'A' ? 'deckA' : 'deckB';
      const updated = [...state[targetDeck]];
      updated[index] = cardId;
      return { [targetDeck]: updated };
    });
    get().triggerPrediction();
  },
  
  triggerPrediction: async () => {
    const { deckA, deckB } = get();
    if (deckA.every(c => c !== null) && deckB.every(c => c !== null)) {
      set({ prediction: { ...get().prediction, loading: true } });
      const res = await callPredictAPI(deckA, deckB);
      set({ prediction: { probability: res.win_probability, band: res.categorical_band, loading: false } });
    }
  }
}));
```

---

## 29. Future-Proofing (Metadata-Driven Card Catalog)

Adding a new card (such as `Ronin`) must require **only** updating configuration files, with zero frontend UI rebuilds:
1.  **JSON Registration**: Add the new card ID, name, elixir, rarity, and image asset path to `features/card_library.json`.
2.  **Metadata Registration**: Add card attributes (combat hp, dps levels, win condition flags) to the card knowledge database configuration file.
3.  The frontend dynamically loads and renders the updated grid and features comparison widgets based solely on this metadata.

---

## 30. Performance Guidelines

*   **Virtual Scrolling**: Render selection menu card lists dynamically using lazy load or scroll offsets to prevent page stuttering.
*   **Image Optimization**: Compress card thumbnail assets into modern, WebP formats to reduce bandwidth.
*   **In-Memory API Caching**: FastAPI caches prediction keys for identical deck matchups, returning cached responses instantly.

---

## 31. Security Considerations

*   **Clipboard Sanitization**: Strip non-alphanumeric symbols and escape parsed text strings during paste actions to prevent Cross-Site Scripting (XSS) injections.
*   **Strict Array Validation**: Discard pasted strings if they cannot be parsed into an array of exactly 8 valid integer card IDs.

---

## 32. Testing Strategy

*   **Unit Tests**: Validate permutation-invariant average pooling, card selection lists, and safe clipboard parsing helper scripts.
*   **Contract Tests**: Mock FastAPI `/predict` REST outputs to verify state changes inside the prediction panel.

---

## 33. Deployment Strategy

*   **Frontend**: Hosted on statically served hosting services (such as Vercel or Netlify).
*   **Backend API**: Hosted on dockerized cloud service containers (such as AWS ECS or GCP Cloud Run).

---

## 34. Version Roadmap

### Version 1.0.0 (Current)
*   FastAPI model inference, side-by-side builder board, collapsed detailed analysis panel, and Tabbed Deck Library.

### Version 2.0.0
*   Interactive 2D card embeddings visualizer using `/embeddings` coordinate metadata, local session caching, and Clash Royale deck link parser.

### Version 3.0.0
*   Graph Neural Network (GNN) model integration, active deck counter suggestions, and player tag sync.

---

## 35. Known Risks

*   **API Availability**: If the backend server goes offline, prediction features will freeze. Provide fallback message warnings.
*   **Clipboard Access Failures**: Strict browser permissions will block clipboard reads. Maintain a manual paste textarea fallback window.

---

## 36. Open Questions

1.  **CORS Rule Restrictions**: Ensure backend CORS policies are updated to accept requests from production frontend hosting domains.
2.  **Card Leveling Variables**: Should V1 support choosing card levels, or assume max level (tournament standards) for all deck comparison runs?

---

## 37. Implementation Checklist

- [ ] Load and verify `Outfit` font family.
- [ ] Implement CSS color variables and spacing variables.
- [ ] Create `DeckSlot` layout with gold rarity frames.
- [ ] Setup Card Selection modal with search filtering.
- [ ] Implement Zustand deck state store.
- [ ] Bind `/predict` POST network calls.
- [ ] Create collapsed comparative progress bars for analysis.
- [ ] Bind Tabbed Deck Library copy trigger.
- [ ] Verify WCAG contrast levels.
- [ ] Run test suite.

---

## 38. Definition of Done

*   All V1 functional features (build, copy, paste, library copy, predictions) are active.
*   Prediction Panel is minimal and detailed analysis is collapsed by default.
*   No custom flat icons are used for cards; only official Clash Royale card thumbnails and rarity borders are rendered.
*   Adding a card requires only updating metadata configurations.
*   Unit tests are green, and page loads in less than 2 seconds.
