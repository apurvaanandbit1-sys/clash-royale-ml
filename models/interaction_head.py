import torch
import torch.nn as nn

class BradleyTerryHead(nn.Module):
    """
    Antisymmetric interaction head for Bradley-Terry comparison.
    Calculates logit matchup score: theta(A, B) = v_A^T W v_B
    where W = M - M^T is guaranteed to be skew-symmetric by design.
    This enforces:
      1. theta(A, A) = 0 => P(A, A) = 0.5 exactly
      2. theta(B, A) = -theta(A, B) => P(A, B) + P(B, A) = 1.0 exactly
    """
    def __init__(self, projection_dim: int, init_scale: float = 0.02):
        super().__init__()
        # Learnable matrix M
        self.M = nn.Parameter(torch.randn(projection_dim, projection_dim) * init_scale)

    def forward(self, v_A: torch.Tensor, v_B: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Parameters:
            v_A: Tensor of shape [Batch, D] (deck A embedding)
            v_B: Tensor of shape [Batch, D] (deck B embedding)
        Returns:
            Tensor of shape [Batch, 1] (matchup logits)
        """
        # Form skew-symmetric matrix W
        W = self.M - self.M.t()
        
        # Batch matrix multiplication: v_B @ W.t() (equivalent to W @ v_B for each batch)
        # Shape: [Batch, D]
        W_v_B = torch.matmul(v_B, W.t())
        
        # Element-wise product followed by row summation
        # Shape: [Batch, 1]
        logits = torch.sum(v_A * W_v_B, dim=1, keepdim=True)
        return logits


class SkillDifference(nn.Module):
    """
    Skill module mapping trophy differences to a win probability logit adjustment.
    At inference (under trophies difference = 0), this bias term evaluates to 0.0,
    leaving only the pure deck intrinsic matchup advantage.
    """
    def __init__(self, hidden_dim: int = 8, scaling_factor: float = 0.001):
        super().__init__()
        self.scaling_factor = scaling_factor
        
        # Simple MLP to map 1D scalar difference to logit adjustment
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize bias to zero to ensure zero input outputs close to 0
        self._init_weights()

    def _init_weights(self):
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, trophy_diff: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Parameters:
            trophy_diff: Tensor of shape [Batch, 1] or [Batch]
        Returns:
            Tensor of shape [Batch, 1] (skill bias logits)
        """
        # Ensure correct shape: [Batch, 1]
        if trophy_diff.dim() == 1:
            td = trophy_diff.unsqueeze(1)
        else:
            td = trophy_diff
            
        # Scale to prevent gradient explosion and keep inputs in reasonable MLP range
        td_scaled = td * self.scaling_factor
        
        # Enforce odd function by design: f(x) = (h(x) - h(-x)) / 2
        # This guarantees f(-x) = -f(x) and f(0) = 0 exactly.
        out_pos = self.mlp(td_scaled)
        out_neg = self.mlp(-td_scaled)
        return 0.5 * (out_pos - out_neg)
