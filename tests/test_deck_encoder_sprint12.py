import unittest
import yaml
import torch
import numpy as np
import tempfile
from pathlib import Path
from models.deck_encoder import DeckEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class TestDeckEncoderSprint12(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        
        self.num_cards = 50
        self.embedding_dim = 16
        self.projection_dim = 16
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_deck_encoder_shapes_and_pooling(self):
        # Test mean pooling
        mean_encoder = DeckEncoder(
            num_cards=self.num_cards,
            embedding_dim=self.embedding_dim,
            pooling_type="mean",
            projection_dim=self.projection_dim
        )
        
        # Test sum pooling
        sum_encoder = DeckEncoder(
            num_cards=self.num_cards,
            embedding_dim=self.embedding_dim,
            pooling_type="sum",
            projection_dim=self.projection_dim
        )
        
        # Mock batch of 5 decks
        x = torch.randint(0, self.num_cards, (5, 8))
        
        mean_out = mean_encoder(x)
        sum_out = sum_encoder(x)
        
        # Verify shapes: [Batch, ProjectionDim]
        self.assertEqual(mean_out.shape, (5, self.projection_dim))
        self.assertEqual(sum_out.shape, (5, self.projection_dim))
        self.assertEqual(mean_out.dtype, torch.float32)

    def test_permutation_invariance(self):
        """
        Critical Requirement: Changing the order of cards in a deck 
        must NOT change the resulting deck embedding.
        """
        encoder = DeckEncoder(
            num_cards=self.num_cards,
            embedding_dim=self.embedding_dim,
            pooling_type="mean",
            projection_dim=self.projection_dim
        )
        encoder.eval()
        
        # Create a single deck
        deck = [1, 5, 8, 12, 19, 23, 27, 34]
        shuffled_deck = [34, 19, 12, 27, 5, 23, 8, 1]
        
        x = torch.tensor([deck], dtype=torch.long)
        x_shuffled = torch.tensor([shuffled_deck], dtype=torch.long)
        
        with torch.no_grad():
            out1 = encoder(x)
            out2 = encoder(x_shuffled)
            
        # Assert they are numerically identical within tolerance
        np.testing.assert_allclose(out1.numpy(), out2.numpy(), rtol=1e-5, atol=1e-5)

    def test_gradient_propagation_to_embeddings(self):
        encoder = DeckEncoder(
            num_cards=self.num_cards,
            embedding_dim=self.embedding_dim,
            pooling_type="mean",
            projection_dim=self.projection_dim
        )
        
        x = torch.randint(0, self.num_cards, (4, 8))
        out = encoder(x)
        
        # Dummy loss: minimize sum of embeddings
        loss = out.sum()
        loss.backward()
        
        # Check that embeddings received gradients
        self.assertIsNotNone(encoder.embeddings.weight.grad)
        grad_norm = encoder.embeddings.weight.grad.norm().item()
        self.assertGreater(grad_norm, 0.0)

    def test_checkpoint_serialization(self):
        encoder = DeckEncoder(
            num_cards=self.num_cards,
            embedding_dim=self.embedding_dim,
            pooling_type="mean",
            projection_dim=self.projection_dim
        )
        encoder.eval()
        
        x = torch.randint(0, self.num_cards, (2, 8))
        with torch.no_grad():
            orig_out = encoder(x).numpy()
            
        # Save state dict
        filepath = Path(self.temp_dir.name) / "encoder_checkpoint.pt"
        torch.save(encoder.state_dict(), filepath)
        
        # Load state dict
        new_encoder = DeckEncoder(
            num_cards=self.num_cards,
            embedding_dim=self.embedding_dim,
            pooling_type="mean",
            projection_dim=self.projection_dim
        )
        new_encoder.load_state_dict(torch.load(filepath))
        new_encoder.eval()
        
        with torch.no_grad():
            loaded_out = new_encoder(x).numpy()
            
        # Assert outputs match exactly
        np.testing.assert_array_equal(orig_out, loaded_out)

    def test_configuration_loading(self):
        # Load from YAML
        with open(PROJECT_ROOT / "config" / "deck_encoder_config.yaml", "r") as f:
            config = yaml.safe_load(f)
            
        encoder = DeckEncoder(
            num_cards=self.num_cards,
            embedding_dim=config["model"]["embedding_dim"],
            pooling_type=config["model"]["pooling_type"],
            hidden_dim=config["model"]["hidden_dim"],
            projection_dim=config["model"]["projection_dim"],
            dropout=config["model"]["dropout"],
            use_layernorm=config["model"]["use_layernorm"],
            init_type=config["model"]["init_type"]
        )
        
        self.assertEqual(encoder.embeddings.embedding_dim, config["model"]["embedding_dim"])
        self.assertEqual(encoder.pooling_type, config["model"]["pooling_type"])

if __name__ == "__main__":
    unittest.main()
