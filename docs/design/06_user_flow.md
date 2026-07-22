# User Flow Documentation
## Clash Royale Matchup Predictor — Version 1.0.0

---

## 1. Landing Page Initialization Flow

*   **Trigger**: User navigates to the application URL.
*   **Actions**:
    1.  Frontend app mounts the DOM.
    2.  Fetches `features/card_library.json` list.
    3.  Pings backend `/health` endpoint to check if the ONNX runtime model is online.
    4.  Initializes Decks A and B with 8 empty slots (dashed borders).
    5.  Renders the Deck Library panel containing Meta and Popular Decks templates.
*   **Result**: Page ready for interaction.

---

## 2. Deck Building Flow

```
[User clicks Empty Slot]
          │
          ▼
[Open CardGridPicker Modal] ───► [User types in Search Bar] ───► [Filter card icons]
          │
          ▼
[User clicks Card Icon]
          │
          ▼
[Is Card already in current Deck?]
     ├── Yes ──► [Show validation warning (No duplicates)]
     └── No  ──► [Add Card ID to Slot State]
          │
          ▼
[Close Modal & Re-render slots]
```

---

## 3. Copy Flow (Active Deck)

*   **Trigger**: User clicks the **Copy** button above Deck A or Deck B.
*   **Actions**:
    1.  App fetches the 8 card IDs from the respective deck state.
    2.  Serializes IDs into standard JSON payload format.
    3.  Attempts to write the string to the system clipboard via `navigator.clipboard.writeText`.
    4.  If successful, triggers temporary Copy feedback button state ("Copied!") for 1.5 seconds.
    5.  If browser blocks access, displays clipboard error message: "Clipboard access rejected. Please copy manually."
*   **Result**: Deck saved to user's system clipboard.

---

## 4. Paste Flow (Active Deck)

*   **Trigger**: User clicks the **Paste** button above Deck A or Deck B.
*   **Actions**:
    1.  App calls `navigator.clipboard.readText()`.
    2.  Attempts to parse text as JSON.
    3.  **Validation checks**:
        *   Is it valid JSON?
        *   Does it contain an array of exactly 8 elements?
        *   Are all elements numbers?
        *   Are all numbers valid card IDs in `card_library.json`?
    4.  If validation passes, updates deck state and triggers a border flash animation on the 8 card slots.
    5.  If validation fails, aborts the update and launches a temporary banner alert explaining the input formatting error.
*   **Result**: Target deck populated from clipboard data.

---

## 5. Prediction Flow

*   **Trigger**: The state changes to 8 complete card IDs in *both* Deck A and Deck B.
*   **Actions**:
    1.  Frontend locks deck slots to prevent edits during calculation.
    2.  Triggers loading spinner inside the central Prediction Panel.
    3.  Sends POST request to backend `/predict` containing:
        *   `p1_deck` (Deck A card IDs).
        *   `p2_deck` (Deck B card IDs).
        *   `trophy_diff: 0.0` (to isolate pure deck-on-deck matchup).
    4.  On receiving API response:
        *   Unlocks slots.
        *   Hides loading spinner.
        *   Updates `winProbability` state.
        *   Displays Favored Deck highlight and Confidence badge.
        *   Renders the Detailed Deck Analysis comparison table below.
*   **Result**: Prediction results displayed to the user.

---

## 6. Deck Library Copy & Paste Flow

*   **Trigger**: User browses the Deck Library and identifies a deck (e.g., "Golem Night Witch").
*   **Actions**:
    1.  User clicks the **Copy** button on the target library deck card.
    2.  App writes the deck's pre-configured card IDs to the system clipboard.
    3.  Button displays confirmation checkmark and "Copied!" text.
    4.  User scrolls to Deck A or Deck B comparison columns.
    5.  User clicks the **Paste** action button on their chosen slot column.
    6.  The target deck loads the copied library cards.
*   **Result**: Pre-built deck loaded into the active workspace.

---

## 7. Error Flow

### API Timeout / Server Down
*   If `/predict` fails to respond or returns 500 error:
    1.  App hides loading spinner.
    2.  Unlocks slots to allow modification.
    3.  Displays error banner: "Prediction Error: Backend server failed to respond. Please try again later."
    4.  Hides Prediction Panel outputs.

### Clipboard Access Blocked
*   If clipboard read permission is rejected by the browser:
    1.  App detects rejection.
    2.  Launches a modal fallback text area input: "Please paste your deck data string here manually."
    3.  User inputs string, clicks "Load", and validation executes.

---

## 8. Open Questions
1.  **Multiple Modal Paste Actions**: If the user has a Clash Royale link copied (instead of the app's custom JSON format), should the app support parsing it in the paste path, or handle it as an error flow?
2.  **Partial Caching**: If the user edits a single card in a deck, should the app keep the previous prediction visible with a transparent overlay during load state, or clear it instantly?

---

## 9. Future Improvements
1.  **Direct Drag-and-Drop User Flow**: Allow dragging from the library card straight to a deck slot container to bypass clipboard permissions entirely.
2.  **Visual Undo Action**: Provide an "Undo Paste" button if a user accidentally overwrites their custom build.

---

## 10. Implementation Notes
*   Keep user flow alerts clear and readable, avoiding raw programming error outputs (like showing a raw stack trace to the user).
*   Enforce a maximum loading state timeout (e.g. 5 seconds) after which predictions abort and report a timeout error.

---

## 11. Potential Risks
*   **Browser Back Button State Loss**: If a user hits the browser back button, state might clear. Wrap local deck states into route query parameters (e.g., `/?deckA=26000000,...&deckB=...`) to ensure browser navigation preserves deck state.
*   **Clipboard Hijacking**: Verify that copy functions do not inject tracking or malicious code into user clipboards.
