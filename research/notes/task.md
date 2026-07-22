# Task Tracker — Repository Cleanup & Plain-English Update (Completed)

- [x] **F1 — Reorganize into a Clean Top-Level Structure**
  - [x] Created `research/`, `research/scripts/`, `research/results/`, and `research/notes/` directories.
  - [x] Moved all one-off analysis and utility scripts (18 total) into `research/scripts/`.
  - [x] Moved CSV and PNG output results (`learning_curve.csv`, `learning_curve.png`, `sprint12_embeddings_pca.csv`) into `research/results/`.
  - [x] Renamed the technical paper to `docs/paper.md` and compiled `docs/paper.pdf`.
  - [x] Moved all other markdown documents, logs, lists, and task tracking files into `research/notes/`.
  - [x] Fixed all internal imports, path resolution strings, and output file paths across all 18 scripts.
  - [x] Updated all cross-references in `README.md` and other documentation files.
  - [x] Verified that `generate_pdf.py` successfully compiles the paper to `docs/paper.pdf` using the updated paths.
- [x] **F2 — Plain-English Explanation Section**
  - [x] Added "What This Does, In Plain English" section near the top of `README.md` explaining:
    - The core problem (separating deck advantage from player skill).
    - Conceptual explanations of the three architectural components (deck encoder pooling, anti-symmetric matchup head, skill bias un-confounding).
    - The honest framing of the predictive results (calibrated/ranking model vs non-significant accuracy edge, ladder noise context).
- [x] **F3 — Design & Product Documentation Suite**
  - [x] Created `docs/design/` directory containing 7 comprehensive documents:
    1. Product Requirements Document (PRD) (`01_prd.md`)
    2. Design System (`02_design_system.md`)
    3. UI/UX Specification (`03_ui_ux_spec.md`)
    4. Component Specification (`04_component_spec.md`)
    5. Technical Architecture (`05_technical_architecture.md`)
    6. User Flow Documentation (`06_user_flow.md`)
    7. Version Roadmap (`07_roadmap.md`)
  - [x] Standardized all documents to include mandatory sections at the end: Open Questions, Future Improvements, Implementation Notes, and Potential Risks.
  - [x] Aligned documentation with existing codebase features and API shapes (FastAPI routes `/predict`, `/embeddings`, `/health`).
- [x] **F4 — Software Design Review**
  - [x] Conducted a detailed, FAANG-level critique of the 7 design documents.
  - [x] Generated [DesignReview.md](file:///c:/projects/clash-royale-ml/docs/design/DesignReview.md) containing evaluation ratings and recommendations.
- [x] **F5 — User Experience Review**
  - [x] Conducted an in-depth review of the product from usability, cognitive psychology, and accessibility perspectives.
  - [x] Generated [UserExperienceReview.md](file:///c:/projects/clash-royale-ml/docs/design/UserExperienceReview.md) detailing scores and critical issues.
- [x] **F6 — Master Specification Consolidation**
  - [x] Consolidated all 7 design documents, `DesignReview.md`, and `UserExperienceReview.md` into a single file.
  - [x] Generated [MASTER_SPECIFICATION.md](file:///c:/projects/clash-royale-ml/docs/design/MASTER_SPECIFICATION.md) covering exactly 38 required sections.
  - [x] Incorporated all latest product decisions: minimal predictions, collapsed-by-default detailed breakdown, gamified progress/comparison widgets, official Clash Royale card thumbnails/rarity frames, and metadata-driven card catalog extensibility.
- [x] **Final Verification & Test Suite**
  - [x] Ran full unit test suite (29/29 tests passing 100% green).
  - [x] Verified clean root directory containing only production files and clean `docs/` and `research/` folders.
