# Design System & UI Components Specification
## Clash Royale Matchup Predictor — Version 1.0.0

---

## 1. Design Philosophy

The Clash Royale Matchup Predictor features a premium, responsive dark mode interface designed for gaming enthusiasts and analytical coaches. The aesthetic borrows cues from high-fidelity gaming interfaces, using dark glassmorphism, glowing accents, and crisp typography to create a clean, modern user experience.

---

## 2. Typography

*   **Font Family**: `Outfit` (Primary Sans-serif) as the default font. Fallback to `Inter` or system sans-serif.
*   **Scale**:
    *   **H1 (Title)**: `32px` (2rem) | Bold (700) | Tracking: -0.02em
    *   **H2 (Headers)**: `24px` (1.5rem) | Semi-Bold (600) | Tracking: -0.01em
    *   **H3 (Sub-headers)**: `18px` (1.125rem) | Medium (500)
    *   **Body (Default)**: `14px` (0.875rem) | Regular (400) | Line-Height: 1.5
    *   **Caption/Labels**: `12px` (0.75rem) | Medium (500) | Line-Height: 1.2

---

## 3. Color Palette

The color palette is built using tailorable CSS custom properties supporting a sleek dark mode. Colors reflect competitive stats (green for positive win rates, red for negative, gold for hero elements).

```css
:root {
  /* Neutral Palette */
  --bg-dark-base: #0f111a;      /* Deep obsidian base */
  --bg-dark-surface: #171926;   /* Glass surface card */
  --bg-dark-elevated: #222538;  /* Popups and inputs */
  --border-light: #2c2f47;      /* Subtle grid separator */
  --border-focus: #4b52ff;      /* Active selection halo */
  
  /* Text Colors */
  --text-primary: #f8fafc;
  --text-secondary: #94a3b8;
  --text-muted: #64748b;
  
  /* Brand & Status Colors */
  --brand-primary: #3b82f6;     /* Electric Blue */
  --brand-accent: #f59e0b;      /* Champion Gold */
  --status-success: #10b981;    /* Favored Green */
  --status-danger: #ef4444;     /* Unfavored Red */
  --status-warning: #f59e0b;    /* Mid-range Elixir */
  
  /* Opacities & Glass */
  --glass-bg: rgba(23, 25, 38, 0.7);
  --glass-blur: blur(12px);
}
```

---

## 4. Spacing Scale

Consistent mathematical increments based on an 8px grid:
*   `4px` (xs): Tight padding, icon offsets.
*   `8px` (sm): Card gap, small button padding.
*   `16px` (md): Standard block padding, grid gap.
*   `24px` (lg): Card padding, section spacing.
*   `32px` (xl): Section gaps, header margin.

---

## 5. UI Elements & Components

### Buttons
*   **Primary Action**: Glow outline on hover, solid fill on click. Electric Blue background.
*   **Secondary Action (Build/Copy/Paste)**: Subtle border, changes background opacity to 10% on hover.
*   **Disabled State**: Grey text and border, cursor not-allowed, opacity 50%.

### Cards
*   **Card Slots**: 8-slot container for decks. When empty, shows a dashed border with a centered plus (+) icon.
*   **Card Picker Items**: Rendered inside the card builder modal. Highlighted with a gold border when selected.

### Inputs (Card Selector Search)
*   Standard text inputs with rounded borders (`8px`), featuring a search icon prefix. 
*   Active state adds a `1px solid var(--border-focus)` border and a subtle blue shadow.

### Icons
*   Use standard, minimal stroke icon libraries (e.g. Lucide or Heroicons).
*   **Build**: Hammer / Edit icon.
*   **Copy**: Clipboard / Dual Card icon.
*   **Paste**: Inverted Arrow / Clipboard icon.

---

## 6. Animations & Transitions

*   **Modal Open/Close**: Smooth fade-in and scale-up transition over `200ms` using `cubic-bezier(0.16, 1, 0.3, 1)`.
*   **Card Hover**: 3D scale transition (`transform: scale(1.05)`) over `150ms`.
*   **Prediction Load**: Fades from 50% opacity to 100% when backend computations complete.

---

## 7. Responsive Philosophy

*   **Mobile (<640px)**: Stacked single-column card grids. The Deck Library is placed below the two-deck comparison board.
*   **Tablet (640px - 1024px)**: Dual columns side-by-side but with a smaller card size grid.
*   **Desktop (>1024px)**: Full side-by-side display with floating prediction results panel in the center.

---

## 8. Accessibility Standards

*   **Contrast**: Text-to-background contrast ratio must adhere to WCAG 2.1 AA requirements (4.5:1 ratio).
*   **Keyboard Navigation**: Modals must trap focus. Users can navigate the card library using `Tab` and select cards with `Enter`/`Space`.
*   **Screen Readers**: Elements like slots, action buttons, and card images must have descriptive `aria-label` tags (e.g. `aria-label="Add card to slot 1 of Deck A"`).

---

## 9. Open Questions
1.  **Card Evolution Indicators**: Should evolved cards feature a special animated frame (e.g., pink shards) to distinguish them from standard cards in the deck builder?
2.  **Sound Effects**: Should action triggers (like selecting cards or calculating predictions) feature subtle sound effects, and if so, how should they be toggled off?

---

## 10. Future Improvements
1.  **Light Mode Theme**: Add a toggle switch to transition the entire layout to a light, clean, high-contrast theme.
2.  **Card Animation Support**: Support animated GIF/WebP files for hero cards in the selection menu.

---

## 11. Implementation Notes
*   Utilize CSS custom properties for all style variables to simplify theme maintenance and future modifications.
*   Enforce a strict layout boundaries rule to prevent resizing cards from shifting adjacent layout grids.

---

## 12. Potential Risks
*   **Render Performance on Low-End Mobile Devices**: Rendering 122 high-resolution card images simultaneously inside the builder modal can cause scrolling lag. Resolve this by utilizing lazy loading or virtual scrolling lists.
*   **Contrast in Feedback States**: Dark green or red backgrounds must be sufficiently dark to keep overlay white text readable.
