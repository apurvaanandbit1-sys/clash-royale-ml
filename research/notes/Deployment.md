# Production Deployment Roadmap

This document outlines the deployment roadmap for integrating the `MatchupModel` into live Matchmaking or deck recommendations.

## 1. Inference Latency & Optimization
*   **Parameters**: The model has only 3,369 weights, resulting in a negligible memory footprint ($<100\text{ KB}$).
*   **ONNX Export**: The PyTorch model can be exported to ONNX to perform sub-millisecond CPU evaluations:
    ```python
    import torch
    # Export Deck Encoder + BT head
    dummy_p1 = torch.zeros((1, 8), dtype=torch.long)
    dummy_p2 = torch.zeros((1, 8), dtype=torch.long)
    dummy_td = torch.zeros((1, 1), dtype=torch.float32)
    torch.onnx.export(model, (dummy_p1, dummy_p2, dummy_td), "matchup_model.onnx")
    ```

## 2. API Integration
The model can be wrapped in a FastAPI service that exposes:
*   `POST /predict_matchup`: Receives two lists of 8 card IDs, and returns the intrinsic matchup probability (setting $\Delta T = 0$ internally).
