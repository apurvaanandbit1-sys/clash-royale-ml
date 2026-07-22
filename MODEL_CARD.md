# Model Card: Siamese Bradley-Terry MatchupModel

## Model Details
*   **Developed by**: [AUTHOR NAME]
*   **Model Type**: Siamese Neural Network with Bradley-Terry Logits Head and Skill Bias Un-confounding
*   **Language(s)**: Python / PyTorch
*   **License**: MIT License

## Intended Use
*   **Primary Use Case**: Estimate the intrinsic matchup advantage between two decks of 8 cards in Clash Royale.
*   **Target Users**: Game design balance teams, competitive tournament organizers, and deck recommendation engines.
*   **Out of Scope**: Real-time battle placement predictions or active card cycling/rotation analysis.

## Inputs & Outputs
*   **Inputs**:
    *   `Deck A`: 8 continuous integer indices representing the cards in Deck A.
    *   `Deck B`: 8 continuous integer indices representing the cards in Deck B.
    *   `Trophy Difference`: Relative trophy difference between Player A and Player B. (Set to 0 at inference).
*   **Outputs**:
    *   `Win Probability`: A float value in the range `[0.0, 1.0]` indicating the probability that Deck A defeats Deck B.

## Training Data & Evaluation
*   **Training Dataset**: 100,542 matches parsed from competitive ladder play.
*   **Evaluation Metrics**: Accuracy (58.12%), ROC-AUC (62.62%), Expected Calibration Error (4.04%), Brier Score (0.2373).

## Limitations & Known Failures
*   **Mirror Matchups**: Decks sharing $\ge 6$ card IDs result in prediction accuracy of **45.00%** (below chance), as card rotation order and tactical placement are not modeled.
*   **Embedding coordinate drift**: Embeddings rotate arbitrarily across different training seeds due to average pooling rotational invariance.
