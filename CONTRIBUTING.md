# Contributing to Clash Royale ML Matchup Predictor

First off, thank you for considering contributing to this project!

## How to Contribute
1.  **Report Bugs**: Open an issue describing the bug, steps to reproduce, and environment details.
2.  **Suggest Improvements**: Suggest enhancements, doc clarifications, or optimization tools.
3.  **Pull Requests**:
    *   Fork the repository and create your branch from `main`.
    *   Make sure all unit tests pass:
        ```bash
        python -m unittest discover tests
        ```
    *   Adhere to a strict architecture freeze. Modifications to network layers, embedding pooling logic, or loss functions are not permitted.
