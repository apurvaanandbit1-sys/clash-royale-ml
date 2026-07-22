import torch
import torch.nn as nn
import torch.nn.init as init

class DeckEncoder(nn.Module):
    """
    Permutation-invariant PyTorch module for encoding Clash Royale decks.
    Pipeline:
      Card IDs [Batch, 8]
        ↓
      Embedding Lookup [Batch, 8, EmbeddingDim]
        ↓
      Shared Card Projection MLP [Batch, 8, ProjectionDim]
        ↓
      Symmetric Pooling (Mean/Sum) [Batch, ProjectionDim]
    """
    def __init__(
        self,
        num_cards: int,
        embedding_dim: int = 16,
        pooling_type: str = "mean",
        hidden_dim: int = 32,
        projection_dim: int = 16,
        dropout: float = 0.1,
        use_layernorm: bool = True,
        init_type: str = "xavier_uniform"
    ):
        super().__init__()
        self.pooling_type = pooling_type.lower()
        if self.pooling_type not in ["mean", "sum"]:
            raise ValueError(f"Unsupported pooling type: {pooling_type}. Choose 'mean' or 'sum'.")

        # 1. Trainable Embedding Layer
        self.embeddings = nn.Embedding(num_cards, embedding_dim)
        
        # Initialize Embeddings
        if init_type == "xavier_uniform":
            init.xavier_uniform_(self.embeddings.weight)
        elif init_type == "xavier_normal":
            init.xavier_normal_(self.embeddings.weight)
        else:
            raise ValueError(f"Unsupported initialization type: {init_type}")

        # 2. Shared Card Projection Network (MLP)
        layers = []
        layers.append(nn.Linear(embedding_dim, hidden_dim))
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, projection_dim))
        
        self.shared_mlp = nn.Sequential(*layers)

        # Initialize shared projection layers
        self._init_weights()

    def _init_weights(self):
        """Applies Xavier initialization to the shared projection MLP linear layers."""
        for module in self.shared_mlp.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Parameters:
            x: Tensor of shape [Batch, 8], dtype torch.long
        Returns:
            Tensor of shape [Batch, ProjectionDim], dtype torch.float32
        """
        # Lookup card embeddings: [Batch, 8] -> [Batch, 8, EmbeddingDim]
        emb = self.embeddings(x)
        
        # Apply shared projection MLP: [Batch, 8, EmbeddingDim] -> [Batch, 8, ProjectionDim]
        proj = self.shared_mlp(emb)
        
        # Permutation-invariant Pooling along the card dimension (dim=1)
        if self.pooling_type == "mean":
            out = proj.mean(dim=1)
        elif self.pooling_type == "sum":
            out = proj.sum(dim=1)
            
        return out
