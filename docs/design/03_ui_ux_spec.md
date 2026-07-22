# UI/UX Specification
## Clash Royale Matchup Predictor — Version 1.0.0

---

## 1. Single-Page Application (SPA) Grid Layout

The application operates as a single-page application structured on a responsive grid.

```
+--------------------------------------------------------+
|                      [HEADER]                          |
|    Clash Royale Intrinsic Matchup Predictor            |
+--------------------------------------------------------+
|                                                        |
|  +------------------+            +------------------+  |
|  |     DECK A       |            |     DECK B       |  |
|  | [Build/Copy/Paste] |            | [Build/Copy/Paste] |  |
|  |  [8 Card Slots]  |            |  [8 Card Slots]  |  |
|  +------------------+            +------------------+  |
|                                                        |
|                   +----------------+                   |
|                   |   PREDICTION   |                   |
|                   |     PANEL      |                   |
|                   +----------------+                   |
|                                                        |
|                   +----------------+                   |
|                   |  COLLAPSED     |                   |
|                   |  DECK ANALYSIS |                   |
|                   +----------------+                   |
|                                                        |
|  +--------------------------------------------------+  |
|  |                  DECK LIBRARY                    |  |
|  |          [Meta Decks] | [Popular Decks]          |  |
|  +--------------------------------------------------+  |
+--------------------------------------------------------+
```

---

## 2. Interactive Sections & Specs

### Section A: Header
*   **Elements**: App Logo, Main Title, Subtitle explaining "Intrinsic Matchup Advantage (Equal Player Skill)".
*   **Status Indicator**: A small dot in the corner showing server connection status (`Green` for active/healthy, `Red` for disconnected/error).

### Section B: Deck Selection Board (Deck A & Deck B)
*   **Deck Container**: Rounded cards with a subtle border.
*   **Action Bar**: Placed directly above the card slots:
    *   **Build**: Trigger button with hammer icon. Focus opens Card Picker Modal.
    *   **Copy**: Trigger button with clipboard icon. Copies JSON list of card IDs to clipboard.
    *   **Paste**: Trigger button with paste icon. Reads clipboard contents and populates the slots.
*   **Card Slots**: An 8-slot flex wrap container.
    *   **Empty Slot State**: Gray background, dashed border, centered icon.
    *   **Filled Slot State**: High-quality card image rendered inside. Hovering a card shows a red delete (x) badge to allow removal.

### Section C: Prediction Results Panel
*   **Display Logic**: Only renders when *both* Deck A and Deck B contain exactly 8 valid cards.
*   **Layout**: Placed centrally between or above the two decks.
*   **Visual Indicators**:
    *   **Favored Header**: Large indicator showing "DECK A FAVORED" or "DECK B FAVORED".
    *   **Win Rate Meter**: Radial gauge or large bold percentage text showing the exact prediction (e.g. `58.1%`).
    *   **Confidence Badge**: Renders the categorical band colored by status (green for strongly favored, amber for even matchup, red for strongly unfavored).

### Section D: Collapsible Detailed Deck Analysis
*   **Display Logic**: Renders below the prediction results.
*   **Layout**: A table comparing the engineered features of Deck A and Deck B side-by-side.
*   **Feature Comparison Layout**:
    *   Left side: Deck A's feature value.
    *   Center column: Feature Label (e.g. "Average Elixir").
    *   Right side: Deck B's feature value.
    *   Visual assistance: Bar graph indicators pointing left and right from the center to indicate strength differences.

### Section E: Deck Library
*   **Tab System**: Horizontal tabs separating categories ("Meta Decks", "Popular Decks").
*   **Deck List**: Vertical scrollable grid containing deck entries.
*   **Deck Entry UI**:
    *   Deck name text.
    *   Average Elixir tag (e.g. `3.2 Avg Elixir`).
    *   Row of 8 mini card images.
    *   **Copy Deck** button. Clicking this button triggers copy feedback ("Copied to clipboard!").

---

## 3. UI States

### 1. Initial / Empty State
*   Deck slots are empty.
*   The Prediction Panel and Collapsible Analysis sections are hidden, showing a prompt: "Please select 8 cards for both decks to see predictions."
*   Deck Library tabs are interactive.

### 2. Loading State
*   Triggered when both decks are filled and the backend API `/predict` is called.
*   A spinner overlay renders over the Prediction Panel. Slots are locked to prevent edits during calculation.

### 3. Error / Unreachable State
*   Triggered if the API fails or clipboard contents cannot be parsed.
*   A banner appears at the top of the interface: "Network Error: Server unreachable. Please check connection and try again." Or: "Invalid Deck: Clipboard data must contain exactly 8 Clash Royale card IDs."

---

## 4. Micro-interactions & Touchpoints

*   **Copy Feedback**: When a user clicks "Copy", the button text shifts to "Copied!" and turns green for 1.5 seconds.
*   **Paste Validation Flash**: When a user clicks "Paste" and the deck loads successfully, the 8 slots execute a quick border fade transition to highlight successful update.
*   **Card Hover Zoom**: Hovering over cards inside the Modal Selection menu triggers a scale up and adds a glow matching the card's rarity (e.g., gold glow for Legendary/Champion).

---

## 5. Open Questions
1.  **Deletion Workflow**: Should clicking a card slot immediately clear it, or should it open the card selection modal with the current card pre-selected for replacement?
2.  **Scroll Position Preservation**: When opening the Card Picker Modal, should the modal preserve the user's last scroll position or reset to the top card selection?

---

## 6. Future Improvements
1.  **Drag and Drop Support**: Allow users to drag a card directly from the Card Picker Modal or Deck Library and drop it into a specific slot in Deck A or Deck B.
2.  **Compare View Toggle**: Add a toggle to stack the two decks vertically on mobile instead of wrapping card slots into two smaller columns.

---

## 7. Implementation Notes
*   Utilize local session variables to preserve deck lists temporarily between browser tab switching.
*   Implement lazy rendering for the Card Picker Modal to ensure the page load time is not impacted by loading 122 images.

---

## 8. Potential Risks
*   **Accidental Paste Execution**: Users clicking the "Paste" button might overwrite their current deck. Resolve this by displaying a temporary confirmation popup if the slot was already populated.
*   **Mobile Screen Width Limits**: Side-by-side tables can easily become clipped on screens smaller than 360px wide. Stack the comparison table rows vertically if the width is constrained.
