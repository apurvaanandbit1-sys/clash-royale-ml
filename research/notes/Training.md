# Training Pipeline

The model training pipeline is implemented in `training/trainer.py` and driven by the entry point `train.py`.

## 1. Loss Function
The model is trained to minimize Binary Cross-Entropy (BCE) Loss:
$$\mathcal{L} = -\frac{1}{B} \sum_{i=1}^{B} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]$$
where $y_i \in \{0, 1\}$ is the actual winner of the match, and $p_i$ is the predicted probability.

## 2. Optimization and Regularization
*   **Optimizer**: AdamW with learning rate $1.0 \times 10^{-3}$ and weight decay $1.0 \times 10^{-4}$ for high regularization.
*   **Gradient Clipping**: Gradients are clipped at a max norm of $1.0$ to prevent exploding gradients.
*   **Learning Rate Scheduler**: `ReduceLROnPlateau` decays the learning rate by a factor of 0.5 if the validation loss does not improve for 1 epoch.

## 3. Checkpoint Manager
The `CheckpointManager` saves checkpoints of model weights, optimizer state, scheduler parameters, and seeds to allow seamless resumption.
*   `best_validation.pt` contains weights corresponding to the lowest validation loss epoch.
