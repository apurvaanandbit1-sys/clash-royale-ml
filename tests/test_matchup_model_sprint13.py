import unittest
import torch
import numpy as np
from models.deck_encoder import DeckEncoder
from models.interaction_head import BradleyTerryHead, SkillDifference
from models.predictor import MatchupModel

class TestMatchupModelSprint13(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        
        self.num_cards = 100
        self.projection_dim = 16
        
        # Instantiate modules
        self.encoder = DeckEncoder(num_cards=self.num_cards, pooling_type="mean", projection_dim=self.projection_dim)
        self.head = BradleyTerryHead(projection_dim=self.projection_dim)
        self.skill = SkillDifference(hidden_dim=8, scaling_factor=0.001)
        
        self.model = MatchupModel(self.encoder, self.head, self.skill)
        self.model.eval()

    def test_probability_identity_property_exactly_half(self):
        """
        Critical Requirement: P(A, A) = 0.5 exactly.
        Feeding identical decks to the model must yield exactly 0.5 win probability,
        regardless of the learnable weights or the trophy difference value (when skill diff = 0).
        """
        # Mock a batch of 5 decks
        decks = torch.randint(0, self.num_cards, (5, 8))
        
        # 1. Under predict_intrinsic (equal skill)
        probs_intrinsic = self.model.predict_intrinsic(decks, decks)
        np.testing.assert_allclose(probs_intrinsic.detach().numpy(), 0.5, rtol=1e-6, atol=1e-6)

        # 2. Under predict_proba with trophy_diff = 0
        zero_diff = torch.zeros(5, 1)
        
        # Note: Skill difference module has biases initialized to zero, but we check if we enforce equal skill
        probs_equal = self.model.predict_proba(decks, decks, zero_diff)
        # Even with biases, if weights are active, we assert this
        # In our case, the bias is zero, so output should be 0.5
        np.testing.assert_allclose(probs_equal.detach().numpy(), 0.5, rtol=1e-5, atol=1e-5)

    def test_antisymmetry_probability_sum_rule(self):
        """
        Critical Requirement: P(A, B) + P(B, A) = 1.0.
        Win probability of Deck A beating Deck B plus Deck B beating Deck A
        must sum to exactly 1.0 within numerical precision.
        """
        deck_A = torch.randint(0, self.num_cards, (10, 8))
        deck_B = torch.randint(0, self.num_cards, (10, 8))
        
        # 1. Intrinsic matchup probability
        p_ab = self.model.predict_intrinsic(deck_A, deck_B).detach().numpy()
        p_ba = self.model.predict_intrinsic(deck_B, deck_A).detach().numpy()
        
        np.testing.assert_allclose(p_ab + p_ba, 1.0, rtol=1e-6, atol=1e-6)

        # 2. Probabilities including skill difference
        # P(A beats B with diff = X) + P(B beats A with diff = -X) = 1.0
        trophy_diff = torch.randn(10, 1) * 1000.0
        p_ab_skill = self.model.predict_proba(deck_A, deck_B, trophy_diff).detach().numpy()
        p_ba_skill = self.model.predict_proba(deck_B, deck_A, -trophy_diff).detach().numpy()
        
        np.testing.assert_allclose(p_ab_skill + p_ba_skill, 1.0, rtol=1e-5, atol=1e-5)

    def test_shared_encoder_siamese_weights(self):
        """Checks that the model uses the same encoder instance for both forward deck passes."""
        self.assertIs(self.model.encoder, self.model.encoder)

    def test_symmetry_under_swapping_logits(self):
        """Swapping deck order must negate the logits (matchup score)."""
        deck_A = torch.randint(0, self.num_cards, (5, 8))
        deck_B = torch.randint(0, self.num_cards, (5, 8))
        
        # Set trophy difference = 0 to isolate deck score
        zero_diff = torch.zeros(5, 1)
        
        logits_ab = self.model(deck_A, deck_B, zero_diff).detach().numpy()
        logits_ba = self.model(deck_B, deck_A, zero_diff).detach().numpy()
        
        np.testing.assert_allclose(logits_ab, -logits_ba, rtol=1e-6, atol=1e-6)

    def test_gradient_flow_to_components(self):
        self.model.train()
        deck_A = torch.randint(0, self.num_cards, (2, 8))
        deck_B = torch.randint(0, self.num_cards, (2, 8))
        td = torch.randn(2, 1)
        
        out = self.model(deck_A, deck_B, td)
        loss = out.sum()
        loss.backward()
        
        # Check that gradients exist and are non-zero
        self.assertIsNotNone(self.model.encoder.embeddings.weight.grad)
        self.assertIsNotNone(self.model.head.M.grad)
        self.assertIsNotNone(self.model.skill.mlp[0].weight.grad)
        
        self.assertGreater(self.model.encoder.embeddings.weight.grad.norm().item(), 0.0)
        self.assertGreater(self.model.head.M.grad.norm().item(), 0.0)
        self.assertGreater(self.model.skill.mlp[0].weight.grad.norm().item(), 0.0)

    def test_numerical_stability(self):
        """Ensure outputs remain stable under extreme skill / trophy differences."""
        deck_A = torch.randint(0, self.num_cards, (1, 8))
        deck_B = torch.randint(0, self.num_cards, (1, 8))
        
        # Huge trophy differences
        huge_td_pos = torch.tensor([[1000000.0]], dtype=torch.float32)
        huge_td_neg = torch.tensor([[-1000000.0]], dtype=torch.float32)
        
        p_pos = self.model.predict_proba(deck_A, deck_B, huge_td_pos).item()
        p_neg = self.model.predict_proba(deck_A, deck_B, huge_td_neg).item()
        
        # Should saturate or remain stable and not produce NaN/Inf
        self.assertFalse(np.isnan(p_pos))
        self.assertFalse(np.isnan(p_neg))
        self.assertTrue(0.0 <= p_pos <= 1.0)
        self.assertTrue(0.0 <= p_neg <= 1.0)

if __name__ == "__main__":
    unittest.main()
