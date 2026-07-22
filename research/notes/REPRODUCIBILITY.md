# Reproducibility Guide

Follow this guide to reproduce the complete experimental sweep, validations, and audits from scratch.

## 1. Environment Setup

*   **Operating System**: Linux / Windows 10+ / macOS
*   **Python Version**: 3.10+
*   **Key Dependencies**:
    *   PyTorch $\ge 2.0$
    *   scikit-learn $\ge 1.2$
    *   pandas $\ge 1.5$
    *   numpy $\ge 1.22$
    *   scipy $\ge 1.9$

---

## 2. Command Pipeline

### Step 2.1: Preprocess Database
Preprocess the SQLite raw database to compile Parquet features:
```bash
python preprocessing/preprocess.py
```

### Step 2.2: Run Unit Tests
Verify all 26 assertions pass:
```bash
python -m unittest discover tests
```

### Step 2.3: Run Hyperparameter Sweep (Sprint 15)
Run coordinate sweeps and write findings:
```bash
python research/scripts/sprint15_experimental_runner.py
```
*Expected Output*: Generated `logs/sprint15_experiments.json`.

### Step 2.4: Run Validation Sweep (Sprint 16)
Train across 10 random seeds and calculate paired t-tests:
```bash
python research/scripts/sprint16_validation_runner.py
```
*Expected Output*: Generated `logs/sprint16_validation.json` showing $p < 1e-19$.

### Step 2.5: Run Code Audit (Sprint 17)
Run data split verification checks:
```bash
python research/scripts/sprint17_audit_runner.py
```
*Expected Output*: Generated `logs/sprint17_audit.json` showing 0 shared instance rows.
