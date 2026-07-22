import numpy as np
from sklearn.metrics import brier_score_loss

def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes the Expected Calibration Error (ECE) for binary classification.
    ECE is the weighted average absolute difference between predicted confidence
    and empirical accuracy in each probability bin.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)

    for i in range(n_bins):
        # Find indices of samples falling into the current bin
        bin_lower = bins[i]
        bin_upper = bins[i + 1]
        
        # Select samples within the current bin interval
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            # Empirical accuracy (win rate) in the bin
            bin_acc = np.mean(y_true[in_bin])
            # Average predicted confidence in the bin
            bin_conf = np.mean(y_prob[in_bin])
            # Weighted absolute difference
            ece += (bin_size / total_samples) * np.abs(bin_acc - bin_conf)

    return float(ece)

def evaluate_calibration(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    """
    Performs comprehensive probability calibration analysis.
    Returns:
        - Brier Score
        - Expected Calibration Error (ECE)
        - Reliability diagram values (confidence, accuracy, and count per bin)
    """
    brier = float(brier_score_loss(y_true, y_prob))
    ece = calculate_ece(y_true, y_prob, n_bins=n_bins)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_stats = []
    
    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]
        
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        bin_size = int(np.sum(in_bin))

        if bin_size > 0:
            bin_acc = float(np.mean(y_true[in_bin]))
            bin_conf = float(np.mean(y_prob[in_bin]))
        else:
            bin_acc = 0.0
            bin_conf = 0.0

        bin_stats.append({
            "bin": f"[{bin_lower:.1f}, {bin_upper:.1f})",
            "samples": bin_size,
            "accuracy": bin_acc,
            "confidence": bin_conf
        })

    return {
        "brier_score": brier,
        "ece": ece,
        "bin_statistics": bin_stats
    }
