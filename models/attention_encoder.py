import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionDeckEncoder(nn.Module):
    """
    Self-Attention Deck Encoder (Set Transformer style).
    Applies multi-head self-attention over the card embeddings set
    before pooling into a fixed 16-D deck representation vector.
    """
    def __init__(self, num_cards: int = 122, embedding_dim: int = 16, num_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_cards = num_cards
        self.embedding_dim = embedding_dim
        
        self.card_embeddings = nn.Embedding(num_cards, embedding_dim)
        nn.init.xavier_uniform_(self.card_embeddings.weight)
        
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(embedding_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        
    def forward(self, card_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            card_ids: Tensor of shape (batch_size, 8)
        Returns:
            v_deck: Tensor of shape (batch_size, 16)
        """
        # Shape: (batch_size, 8, 16)
        x = self.card_embeddings(card_ids)
        
        # Self-Attention over 8 card tokens
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm(x + attn_out)
        
        # Shared MLP projection
        proj = self.mlp(x)
        
        # Permutation-invariant average pooling
        v_deck = torch.mean(proj, dim=1)
        return v_deck
