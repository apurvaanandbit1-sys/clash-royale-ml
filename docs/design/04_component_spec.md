# Component Specification
## Clash Royale Matchup Predictor — Version 1.0.0

---

## 1. DeckBuilderSlot

*   **Purpose**: Represents a single card slot out of the 8 cards in a deck.
*   **Inputs**:
    *   `card_id: Optional[int]`: The ID of the currently loaded card. Defaults to `null`.
    *   `card_name: Optional[str]`: The name of the card.
    *   `image_url: Optional[str]`: The URL of the card's PNG image.
    *   `slot_index: int`: The position of the slot (0 to 7).
*   **Outputs**:
    *   `onClick(slot_index: int)`: Triggers when the slot is clicked (opens selector modal).
    *   `onRemove(slot_index: int)`: Triggers when the user deletes the card from the slot.
*   **States**:
    *   `Empty`: Displays a dashed border, plus (+) icon, and grey background.
    *   `Filled`: Renders the card image, hides the plus icon, and displays a delete (x) button on hover.
*   **Behavior**: Clicking the slot when empty opens the card picker modal. Clicking the delete icon clears the slot state.
*   **Variants**: None.

---

## 2. CardGridPicker (Modal Selector)

*   **Purpose**: A full-screen or modal overlay that allows users to browse and select a card from the Clash Royale card pool.
*   **Inputs**:
    *   `isOpen: bool`: Controls modal visibility.
    *   `selected_cards: List[int]`: Current cards in the active deck (to prevent selecting duplicates).
*   **Outputs**:
    *   `onSelectCard(card_id: int)`: Triggers when a card is clicked, closing the modal.
    *   `onClose()`: Triggers when the close button or background overlay is clicked.
*   **States**:
    *   `Visible`: Modal active on top of UI.
    *   `Search Filtering`: Limits card grid items based on user's query in the search bar.
*   **Behavior**: Focus is locked within the modal. Pressing `Escape` closes the modal. Shows cards grouped by rarity or elixir if desired.
*   **Variants**: None.

---

## 3. DeckLibraryCard

*   **Purpose**: Displays a pre-configured deck entry inside the Deck Library.
*   **Inputs**:
    *   `deck_name: str`: Display name (e.g. "Golem Night Witch").
    *   `card_ids: List[int]`: Array of 8 card IDs.
    *   `average_elixir: float`: Pre-calculated average elixir cost.
*   **Outputs**:
    *   `onCopy(card_ids: List[int])`: Triggers when the user clicks the copy button.
*   **States**:
    *   `Default`: Shows card images, name, elixir, and Copy button.
    *   `Copied`: Temporary active state changing the Copy button style to "Copied!" for 1.5 seconds.
*   **Behavior**: Generates mini card thumbnail images for the 8 cards. Clicking "Copy" copies the deck's card ID list.
*   **Variants**: None.

---

## 4. DeckActionGroup

*   **Purpose**: Row of action buttons located above the deck containers.
*   **Inputs**:
    *   `is_deck_empty: bool`: Indicates if all 8 slots are empty.
    *   `is_deck_full: bool`: Indicates if all 8 slots are populated.
*   **Outputs**:
    *   `onBuild()`: Opens the CardGridPicker modal at Slot 0.
    *   `onCopy()`: Copies current deck state.
    *   `onPaste()`: Reads system clipboard and populates deck slots.
*   **States**:
    *   `Copy Disabled`: Inactive state of copy button when the deck is empty.
*   **Behavior**: Paste checks the clipboard contents. If the parsed array is not equal to 8 integer IDs, it triggers an error toast message.
*   **Variants**: None.

---

## 5. PredictionPanel

*   **Purpose**: Central dashboard displaying matchup results.
*   **Inputs**:
    *   `win_probability: float`: Raw win probability value returned by backend API.
    *   `categorical_band: str`: Human-readable confidence label (e.g. "Slightly Favored").
    *   `is_loading: bool`: Shows loading indicator if true.
*   **Outputs**: None.
*   **States**:
    *   `Hidden`: Inactive state when decks are incomplete.
    *   `Loading`: Displays circular spinner overlay.
    *   `Success`: Displays probability metric, favored color background, and confidence band.
*   **Behavior**: Animates the win probability number rolling up from 0% to the target percentage when loaded.
*   **Variants**:
    *   `Green/Success`: Rendered when Deck A is favored.
    *   `Red/Danger`: Rendered when Deck B is favored (Deck A is unfavored).

---

## 6. DeckFeaturesTable

*   **Purpose**: Side-by-side comparison table of calculated features.
*   **Inputs**:
    *   `deck_a_features: Dict[str, Any]`: Feature map returned by feature engine.
    *   `deck_b_features: Dict[str, Any]`: Feature map returned by feature engine.
*   **Outputs**:
    *   `onToggle()`: Collapses or expands the table view.
*   **States**:
    *   `Collapsed`: Hides table, shows expansion arrow icon.
    *   `Expanded`: Shows full feature list with comparative graphs.
*   **Behavior**: Computes delta differences for properties (e.g., Tankiness difference) and draws progress bar overlays.
*   **Variants**: None.

---

## 7. Open Questions
1.  **Deck Library Componentization**: Should the Deck Library card be lazy-rendered to improve initial viewport page performance?
2.  **Shared Modal State**: Should there be a single global instance of the `CardGridPicker` modal, or should Deck A and Deck B contain their own individual modal wrapper states?

---

## 8. Future Improvements
1.  **Multiple Card Picker layouts**: Provide tabs in the `CardGridPicker` to sort cards by elixir cost, alphabetical name order, or rarity tiers.
2.  **Comparative Feature Highlights**: Highlight the winning features in the comparison table with a subtle background color overlay.

---

## 9. Implementation Notes
*   Implement strict input type validation on `DeckBuilderSlot` properties to prevent runtime rendering crashes if invalid card data is passed.
*   Verify that `PredictionPanel` animations utilize hardware-accelerated transitions to avoid page rendering stuttering.

---

## 10. Potential Risks
*   **Browser Clipboard Limits**: Paste button will fail on browsers that block clipboard API access. A manual text area popup input fallback is needed inside the `DeckActionGroup` component behavior.
*   **Image Failures**: If card thumbnail URLs become broken, render a fallback card element displaying the card name as text.
