# release Checklist - v1.0.0

This checklist must be fully executed before tagging a release of the Clash Royale Intrinsic Matchup Predictor codebase.

## 1. Documentation Checks
- [ ] README.md contains accurate test metrics (Acc: 58.12%, ROC-AUC: 62.62%).
- [ ] MODEL_CARD.md details known failure slices (mirror decks accuracy is 45%).
- [ ] research/notes/Overview.md, research/notes/Architecture.md, and research/notes/Validation.md are current.

## 2. Code & Tests Check
- [ ] All 26 unit tests are green:
  ```bash
  python -m unittest discover tests
  ```
- [ ] Independent split audit completes with 0 instance-level overlap:
  ```bash
  python research/scripts/sprint17_audit_runner.py
  ```

## 3. Deployment Artifacts
- [ ] Pretrained weights are saved at `models/checkpoints/sprint13_matchup_model.pt`.
- [ ] Requirements.txt and environment.yml are locked.
- [ ] License and Code of Conduct files exist.
