# Troubleshooting Guide

### 1. "pin_memory argument is set as true but no accelerator is found"
This is a standard PyTorch warning when running on CPU-only machines. The dataloaders will automatically fall back to standard non-pinned memory and function correctly. You can ignore this warning.

### 2. FileNotFoundError: Parquet file not found
Ensure you execute preprocessing first to compile the parquet features:
```bash
python preprocessing/preprocess.py
```

### 3. KeyError: card_id during training
This occurs if new cards are present in the dataset that are missing from `card_library.json`. Update `data/card_library.json` to include the missing card entry.
