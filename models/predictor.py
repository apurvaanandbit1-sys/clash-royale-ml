import torch
import torch.nn as nn
from models.deck_encoder import DeckEncoder
from models.interaction_head import BradleyTerryHead, SkillDifference

class MatchupModel(nn.Module):
    """
    Top-level Siamese Matchup Model wrapper.
    Shares weights of a single DeckEncoder to encode both Player 1 and Player 2 decks,
    passes the embeddings to the BradleyTerryHead, and adds the SkillDifference logit bias.
    """
    def __init__(
        self,
        encoder: DeckEncoder,
        head: BradleyTerryHead,
        skill: SkillDifference
    ):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.skill = skill

    def forward(
        self, 
        p1_deck: torch.Tensor, 
        p2_deck: torch.Tensor, 
        trophy_diff: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass. Computes training logits (unactivated).
        Parameters:
            p1_deck: Tensor of shape [Batch, 8] (Long)
            p2_deck: Tensor of shape [Batch, 8] (Long)
            trophy_diff: Tensor of shape [Batch, 1] or [Batch] (Float)
        Returns:
            Tensor of shape [Batch, 1] containing raw logit differences.
        """
        # Encode decks using the shared encoder
        v_A = self.encoder(p1_deck)
        v_B = self.encoder(p2_deck)
        
        # Calculate antisymmetric matchup advantage
        theta = self.head(v_A, v_B)
        
        # Calculate skill bias logit adjustment
        gamma = self.skill(trophy_diff)
        
        # Combine logits
        logits = theta + gamma
        return logits

    def predict_proba(
        self, 
        p1_deck: torch.Tensor, 
        p2_deck: torch.Tensor, 
        trophy_diff: torch.Tensor
    ) -> torch.Tensor:
        """Computes win probabilities for Player 1 including skill differences."""
        logits = self.forward(p1_deck, p2_deck, trophy_diff)
        return torch.sigmoid(logits)

    def predict_intrinsic(
        self, 
        p1_deck: torch.Tensor, 
        p2_deck: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the intrinsic matchup advantage (P_win) between Deck A and Deck B
        assuming equal player skill (trophy difference = 0).
        """
        v_A = self.encoder(p1_deck)
        v_B = self.encoder(p2_deck)
        theta = self.head(v_A, v_B)
        return torch.sigmoid(theta)
